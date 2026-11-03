"""Pipeline completo da ablação de remoção de fundo, classe por classe, com relatório HTML.

Para cada classe em ``CONFIG['classes']``:

  1. convert  — dataset sem fundo -> YOLO-seg (``01_build_yolo_dataset.py``)
  2. yolo     — treina YOLO26-seg (``02_train_yolo_seg.py``)
  3. masks    — máscaras no dataset original (``03_generate_masks.py``)
  4. audit    — área mantida × GT perdido por estratégia (``04_audit_masks.py``)
  5. select   — finalistas (``05_select_strategies.py``)
  6. frozen   — braço A: NF head com entrada mascarada, SEDifferNet com fundo congelado (``06``)
  7. retrain  — braço B: retreina o SEDifferNet na imagem mascarada (só nível de imagem; D-24)
                (``07_retrain_encoder_masked.py``)
  8. report   — pixel × referência com fundo para as heads do braço A (``08_report.py``)
  9. image    — nível de imagem: encoder com fundo × sem fundo × retreinado (``09_image_level_compare.py``)

No fim, gera ``<rodada>/relatorio_pipeline.html`` (autocontido: tabelas, gráficos, imagens).

Etapas já concluídas são reaproveitadas (pesos YOLO, máscaras, audit, heads, métricas e
cache), então rodar de novo com ``--run_dir`` retoma de onde parou. Uma classe que falha não
interrompe as outras; o status fica em ``pipeline_status.json`` e no HTML.

Uso (a partir de ``differnet/``):

    python ablation_bg_removal/run_all.py --dry_run                 # mostra o plano
    python ablation_bg_removal/run_all.py                           # tudo, todas as classes
    python ablation_bg_removal/run_all.py --classes vari-grip,yoke-suspension
    python ablation_bg_removal/run_all.py --stages audit,select,report,image --run_dir <rodada>
    python ablation_bg_removal/run_all.py --html_only --run_dir <rodada>
    python ablation_bg_removal/run_all.py --config minha_config.json   # sobrescreve CONFIG

Custo aproximado por classe (RTX 4060): YOLO 100 épocas 25 min (vari) a ~3 h (yoke); cada
NF head 1–2,5 h; SEDifferNet do braço B (~200 épocas) várias horas a mais de um dia (yoke);
relatório e nível de imagem de minutos a ~30 min (yoke).
"""
import argparse
import glob
import json
import os
import shutil
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
DIFFERNET_ROOT = os.path.dirname(HERE)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURAÇÃO — edite aqui (ou passe --config arquivo.json com as mesmas chaves)
# ═══════════════════════════════════════════════════════════════════════════════
CONFIG = {
    # ─── Caminhos ────────────────────────────────────────────────────────────
    # Dataset original (com fundo), layout MVTec: <classe>/{train,test,ground_truth}
    'orig_root': r'C:\Users\teo-s\Documents\GitHub\anomaly-detection-dataset\insplad-seg\insplad-seg',
    # Dataset sem fundo (fundo preto): origem dos rótulos da YOLO e domínio do encoder noBg
    'nobg_root': r'C:\Users\teo-s\Documents\GitHub\anomaly-detection-dataset\insplad-seg-no-back-treino',
    # SEDifferNet treinado COM fundo: <pasta>/<classe>/*.pt (um por classe)
    'bg_encoder_dir': os.path.join(DIFFERNET_ROOT, 'final_models', 'SEDiffernet'),
    # SEDifferNet treinado SEM fundo: <pasta>/<classe>_no_bg_best.pt (um por classe)
    'nobg_encoder_dir': os.path.join(DIFFERNET_ROOT, 'final_models', 'SEDiffernet_noBg'),
    # Rodada de NF heads COM fundo usada como referência de pixel (treino_nf_heads/run_*)
    'reference_head_run': os.path.join(DIFFERNET_ROOT, 'resultado_analise_final', 'treino_nf_heads',
                                       'run_20261025_181523'),
    # Onde criar as rodadas (run_<ts>/) com resultados e o HTML
    'output_root': os.path.join(DIFFERNET_ROOT, 'resultado_analise_final', 'ablacao_mascara_objeto'),
    # Artefatos pesados: dataset YOLO e datasets mascarados, pesos YOLO, máscaras
    'data_dir': os.path.join(HERE, 'data'),
    'yolo_runs_dir': os.path.join(HERE, 'runs'),
    'masks_dir': os.path.join(HERE, 'masks'),
    # Python do ambiente da pesquisa
    'python': r'C:\Users\teo-s\.conda\envs\anomaly_attentideffernet\python.exe',

    # ─── O que rodar ─────────────────────────────────────────────────────────
    'classes': ['glass-insulator', 'lightning-rod-suspension', 'polymer-insulator-upper-shackle',
                'vari-grip', 'yoke-suspension'],
    'stages': ['convert', 'yolo', 'masks', 'audit', 'select', 'frozen', 'retrain', 'report', 'image'],
    'force': [],  # etapas a refazer mesmo se já existirem (ex.: ['yolo', 'masks'])

    # ─── YOLO26-seg ──────────────────────────────────────────────────────────
    'yolo_model': 'yolo26s-seg.pt',
    'yolo_epochs': 100,
    'yolo_batch': 8,
    'yolo_imgsz': 640,
    'yolo_patience': 20,
    'mask_conf': 0.10,

    # ─── Audit / seleção ─────────────────────────────────────────────────────
    'audit_strategies': 'raw,dil8,dil16,dil32,dil64,hull,hull+dil16,bbox10,bbox20',
    'force_strategies': 'dil16',  # D-20; '' = usar a regra pré-registrada de D-19
    'fills': 'imagenet_mean',  # D-21: 'black' removido (FP no contorno em vari-grip)
    'feathers': '0',
    'include_reference_nobg': False,  # True = também raw__black (repete o experimento "sem fundo" preto)

    # ─── NF head / braço B ───────────────────────────────────────────────────
    'recipe': 'general',
    'seed': 42,
    'val_fraction': 0.4,
    'split_seed': 42,
    # Braço B: retreina o SEDifferNet na imagem mascarada (só o flow de imagem muda; D-24). A head de
    # pixel não é retreinada porque sairia idêntica à do braço A; a etapa image compara o Image AUROC.
    # '' = todas as candidatas de finalists.json; 'auto' = só as com Δ val >= 0 no braço A
    # (exige o report do braço A antes); ou lista explícita 'dil16__imagenet_mean,...'
    'arm_b_variants': '',
    'arm_b_encoder_pick': 'best_test',  # igual aos final_models (época pelo Image AUROC no teste); 'latest' = última
    'arm_b_pre_epochs': None,  # None = config.py (smoke test: ex. 1)
    'arm_b_meta_epochs': None,

    # ─── Relatórios ──────────────────────────────────────────────────────────
    'n_boot': 1000,
    'encoder_views': 64,
    'image_cross': True,  # nível de imagem: também bg@nobg e nobg@orig
    'image_on_masked': True,  # nível de imagem: encoder noBg nos datasets mascarados das variantes
    'html_mask_previews': 4,
}

STAGES = ('convert', 'yolo', 'masks', 'audit', 'select', 'frozen', 'retrain', 'report', 'image')
ENV_KEYS = dict(orig_root='ORIG_ROOT', nobg_root='NOBG_ROOT', data_dir='DATA_DIR', yolo_runs_dir='RUNS_DIR',
                masks_dir='MASKS_DIR', output_root='ABLATION_ROOT', reference_head_run='REFERENCE_HEAD_RUN',
                bg_encoder_dir='BG_ENCODER_DIR', nobg_encoder_dir='NOBG_ENCODER_DIR', python='PYTHON')


def apply_env(cfg):
    """Exporta os caminhos para os scripts das etapas (lidos em ``common.py``)."""
    for key, env in ENV_KEYS.items():
        os.environ[f'ABR_{env}'] = str(cfg[key])
    os.environ['PYTHONIOENCODING'] = 'utf-8'


apply_env(CONFIG)  # antes de importar common, para que as constantes já reflitam CONFIG
sys.path.insert(0, HERE)
import common as cm  # noqa: E402


class Pipeline:
    def __init__(self, cfg, run_dir, dry_run=False):
        self.cfg = cfg
        self.run_dir = run_dir
        self.dry_run = dry_run
        self.status_path = os.path.join(run_dir, 'pipeline_status.json')
        self.status = (cm.read_json(self.status_path) if os.path.isfile(self.status_path)
                       else {'classes': {}, 'config': cfg})
        self.status['config'] = cfg

    # ─── infraestrutura ──────────────────────────────────────────────────────
    def script(self, name):
        return os.path.join(HERE, name)

    def run(self, class_name, stage, script, args):
        cmd = [self.cfg['python'], self.script(script)] + [str(a) for a in args]
        if self.dry_run:
            print(f'[dry-run] {class_name:32s} {stage:8s} ' + ' '.join(f'"{c}"' if ' ' in c else c for c in cmd))
            return
        import heads as hd
        log = os.path.join(self.run_dir, 'logs', f'{class_name}__{stage}.txt')
        print(f'\n{"=" * 100}\n{class_name} | {stage} | log: {log}\n{"=" * 100}')
        code, minutes = hd.run_logged(cmd, log)
        self.set(class_name, stage, 'ok' if code == 0 else 'failed', minutes=round(minutes, 2), log=log,
                 returncode=code)
        if code != 0:
            raise RuntimeError(f'{script} terminou com código {code} (log: {log})')

    def set(self, class_name, stage, state, **info):
        entry = self.status['classes'].setdefault(class_name, {})
        entry[stage] = dict(state=state, at=cm.timestamp(), **info)
        if not self.dry_run:
            cm.write_json(self.status_path, self.status)

    def skip(self, class_name, stage, reason):
        print(f'{class_name:32s} {stage:8s} pulado: {reason}')
        if not self.dry_run:
            self.set(class_name, stage, 'skipped', reason=reason)

    def forced(self, stage):
        return stage in self.cfg['force']

    # ─── localização de artefatos ────────────────────────────────────────────
    def yolo_weights(self, class_name):
        cands = glob.glob(os.path.join(self.cfg['yolo_runs_dir'], 'yolo_seg', f'{class_name}_*', 'weights', 'best.pt'))
        return max(cands, key=os.path.getmtime) if cands else None

    def masks_dir(self, class_name):
        cands = [d for d in glob.glob(os.path.join(self.cfg['masks_dir'], f'{class_name}_yolo_*'))
                 if os.path.isfile(os.path.join(d, 'manifest.json'))]
        return max(cands, key=os.path.getmtime) if cands else None

    def class_dir(self, class_name):
        return os.path.join(self.run_dir, class_name)

    # ─── etapas ──────────────────────────────────────────────────────────────
    def stage_convert(self, c):
        if os.path.isfile(os.path.join(self.cfg['data_dir'], 'yolo_seg', c, 'data.yaml')) and not self.forced('convert'):
            return self.skip(c, 'convert', 'dataset YOLO já existe')
        args = ['--classes', c, '--orig_root', self.cfg['orig_root'], '--nobg_root', self.cfg['nobg_root'],
                '--out_root', os.path.join(self.cfg['data_dir'], 'yolo_seg'), '--overwrite']
        self.run(c, 'convert', '01_build_yolo_dataset.py', args)

    def stage_yolo(self, c):
        w = self.yolo_weights(c)
        if w and not self.forced('yolo'):
            return self.skip(c, 'yolo', f'pesos existentes {w}')
        args = ['--classes', c, '--model', self.cfg['yolo_model'], '--epochs', self.cfg['yolo_epochs'],
                '--batch', self.cfg['yolo_batch'], '--imgsz', self.cfg['yolo_imgsz'],
                '--patience', self.cfg['yolo_patience'], '--seed', self.cfg['seed'],
                '--data_root', os.path.join(self.cfg['data_dir'], 'yolo_seg'), '--runs_root', self.cfg['yolo_runs_dir']]
        self.run(c, 'yolo', '02_train_yolo_seg.py', args)

    def stage_masks(self, c):
        m = self.masks_dir(c)
        if m and not self.forced('masks'):
            return self.skip(c, 'masks', f'máscaras existentes {os.path.basename(m)}')
        w = self.yolo_weights(c)
        if not w and not self.dry_run:
            raise FileNotFoundError(f'{c}: sem pesos YOLO; rode a etapa yolo.')
        args = ['--classes', c, '--source', 'yolo', '--conf', self.cfg['mask_conf'], '--imgsz', self.cfg['yolo_imgsz'],
                '--orig_root', self.cfg['orig_root'], '--out_root', self.cfg['masks_dir']]
        if w:
            args += ['--weights', w]
        self.run(c, 'masks', '03_generate_masks.py', args)

    def stage_audit(self, c):
        if os.path.isfile(os.path.join(self.class_dir(c), 'audit', 'audit_summary.csv')) and not self.forced('audit'):
            return self.skip(c, 'audit', 'audit existente nesta rodada')
        m = self.masks_dir(c)
        if not m and not self.dry_run:
            raise FileNotFoundError(f'{c}: sem máscaras; rode a etapa masks.')
        args = ['--classes', c, '--run_dir', self.run_dir, '--mask_source', 'yolo', '--orig_root', self.cfg['orig_root'],
                '--strategies', self.cfg['audit_strategies'], '--val_fraction', self.cfg['val_fraction'],
                '--split_seed', self.cfg['split_seed']]
        if m:
            args += ['--masks_dir', m]
        self.run(c, 'audit', '04_audit_masks.py', args)

    def stage_select(self, c):
        if os.path.isfile(os.path.join(self.class_dir(c), 'finalists.json')) and not self.forced('select'):
            return self.skip(c, 'select', 'finalists.json existente')
        args = ['--classes', c, '--run_dir', self.run_dir, '--fills', self.cfg['fills'], '--feathers', self.cfg['feathers']]
        if self.cfg['force_strategies']:
            args += ['--force_strategies', self.cfg['force_strategies']]
        if not self.cfg['include_reference_nobg']:
            args += ['--no_reference']
        self.run(c, 'select', '05_select_strategies.py', args)

    def stage_frozen(self, c):
        args = ['--classes', c, '--run_dir', self.run_dir, '--recipe', self.cfg['recipe'], '--seed', self.cfg['seed'],
                '--skip_existing', '--se_checkpoint', cm.find_class_encoder(self.cfg['bg_encoder_dir'], c)]
        self.run(c, 'frozen', '06_train_nf_head_masked.py', args)

    def stage_retrain(self, c):
        args = ['--classes', c, '--run_dir', self.run_dir, '--recipe', self.cfg['recipe'], '--seed', self.cfg['seed'],
                '--encoder_pick', self.cfg['arm_b_encoder_pick'], '--skip_existing']
        variants = self.cfg['arm_b_variants']
        if variants == 'auto':
            if not os.path.isfile(os.path.join(self.run_dir, 'summary.csv')) and not self.dry_run:
                raise FileNotFoundError("arm_b_variants='auto' exige o report do braço A antes (summary.csv).")
            variants = ''
        elif not variants:
            fin = os.path.join(self.class_dir(c), 'finalists.json')
            if os.path.isfile(fin):
                variants = ','.join(v['variant'] for v in cm.read_json(fin)['variants'] if v['role'] == 'candidate')
            elif not self.dry_run:
                raise FileNotFoundError(f'{c}: sem finalists.json; rode a etapa select.')
            else:
                variants = '<candidatas de finalists.json>'
        if variants:
            args += ['--variants', variants]
        for key, flag in (('arm_b_pre_epochs', '--pre_epochs'), ('arm_b_meta_epochs', '--meta_epochs')):
            if self.cfg[key] is not None:
                args += [flag, self.cfg[key]]
        self.run(c, 'retrain', '07_retrain_encoder_masked.py', args)

    def retrained_encoders(self, c):
        """(variante, checkpoint do SEDifferNet do braço B) das heads B concluídas."""
        import heads as hd
        out = {}
        for h in hd.load_index(self.class_dir(c)).get('heads', []):
            if (h.get('arm') == 'B' and h.get('status') in ('SUCCESS', 'ENCODER_ONLY')
                    and os.path.isfile(h.get('se_checkpoint', ''))):
                out[h['variant']] = h['se_checkpoint']
        return out

    def stage_report(self, c):
        args = ['--classes', c, '--run_dir', self.run_dir, '--reference_run', self.cfg['reference_head_run'],
                '--n_boot', self.cfg['n_boot'], '--encoder_views', self.cfg['encoder_views']]
        if not self.forced('report'):
            args += ['--skip_existing']
        self.run(c, 'report', '08_report.py', args)

    def stage_image(self, c):
        out = os.path.join(self.run_dir, 'image_level', c)
        if os.path.isdir(out) and not self.dry_run:
            shutil.rmtree(out)  # scores ficam em cache; refazer o resumo é barato
        args = ['--classes', c, '--orig_root', self.cfg['orig_root'], '--nobg_root', self.cfg['nobg_root'],
                '--views', self.cfg['encoder_views'], '--val_fraction', self.cfg['val_fraction'],
                '--split_seed', self.cfg['split_seed'], '--n_boot', max(self.cfg['n_boot'], 1000),
                '--out_dir', out, '--cache_dir', os.path.join(self.run_dir, 'image_level', 'cache')]
        if self.cfg['image_cross']:
            args += ['--cross']
        masked = dict(self.masked_datasets(c))
        if self.cfg['image_on_masked']:
            for name, root in masked.items():
                args += ['--extra', f'{name}={root}']
        for variant, ckpt in self.retrained_encoders(c).items():
            data = f'masked_{variant}'
            if data in masked:
                if not self.cfg['image_on_masked']:
                    args += ['--extra', f'{data}={masked[data]}']
                args += ['--encoder', f'retrained_{variant}={ckpt}@{data}']
        self.run(c, 'image', '09_image_level_compare.py', args)

    def masked_datasets(self, c):
        import masking as mk
        fin = os.path.join(self.class_dir(c), 'finalists.json')
        audit_cfg = os.path.join(self.class_dir(c), 'audit', 'audit_config.json')
        if not (os.path.isfile(fin) and os.path.isfile(audit_cfg)):
            return []
        masks_dir = cm.read_json(audit_cfg)['masks_dir']
        out = []
        for v in cm.read_json(fin)['variants']:
            root = mk.masked_dataset_root(masks_dir, v['variant'], self.cfg['data_dir'])
            if os.path.isdir(os.path.join(root, c, 'test')):
                out.append((f'masked_{v["variant"]}', root))
        return out

    # ─── execução ────────────────────────────────────────────────────────────
    def run_class(self, c):
        ok = True
        for stage in STAGES:
            if stage not in self.cfg['stages']:
                continue
            if not ok and stage != 'image':  # nível de imagem não depende das etapas de pixel
                continue
            t = time.time()
            try:
                getattr(self, f'stage_{stage}')(c)
            except Exception as exc:
                if not self.dry_run:
                    entry = self.status['classes'].get(c, {}).get(stage, {})
                    if entry.get('state') != 'failed':
                        self.set(c, stage, 'failed', error=str(exc), traceback=traceback.format_exc(limit=3))
                if stage == 'retrain':  # o braço A ainda pode ser relatado
                    print(f'\n!!! {c}: braço B falhou ({time.time() - t:.0f}s): {exc}\n'
                          f'    report/image seguem só com o braço A.')
                    continue
                print(f'\n!!! {c}: etapa {stage} falhou ({time.time() - t:.0f}s): {exc}\n'
                      f'    etapas de pixel seguintes desta classe foram puladas.')
                ok = False
        return ok


class TeeStream:
    """Escreve no console e em um arquivo de log. ``heads.run_logged`` usa ``.console`` para
    barras de progresso (só no console) e ``.file`` para as linhas finais."""

    def __init__(self, console, file):
        self.console = console
        self.file = file

    def write(self, text):
        self.console.write(text)
        self.file.write(text)
        self.file.flush()
        return len(text)

    def flush(self):
        self.console.flush()
        self.file.flush()

    def isatty(self):
        return self.console.isatty()

    def __getattr__(self, name):
        return getattr(self.console, name)


def start_log_file(run_dir):
    """Duplica stdout/stderr do run_all.py em ``<rodada>/pipeline_log.txt`` (anexa ao retomar)."""
    path = os.path.join(run_dir, 'pipeline_log.txt')
    f = open(path, 'a', encoding='utf-8')
    f.write(f'\n{"#" * 100}\n# {time.strftime("%Y-%m-%d %H:%M:%S")}  {" ".join(sys.argv)}\n{"#" * 100}\n')
    sys.stdout = TeeStream(sys.stdout, f)
    sys.stderr = TeeStream(sys.stderr, f)
    return path


def load_config(path):
    cfg = dict(CONFIG)
    if path:
        with open(path, encoding='utf-8') as f:
            user = json.load(f)
        unknown = set(user) - set(CONFIG)
        if unknown:
            raise KeyError(f'Chaves desconhecidas em {path}: {sorted(unknown)}')
        cfg.update(user)
    return cfg


def validate(cfg):
    problems = []
    for key in ('orig_root', 'nobg_root', 'bg_encoder_dir', 'nobg_encoder_dir', 'reference_head_run', 'python'):
        if not os.path.exists(cfg[key]):
            problems.append(f'{key}: não existe -> {cfg[key]}')
    bad = [s for s in cfg['stages'] if s not in STAGES]
    if bad:
        problems.append(f'etapas desconhecidas: {bad}; válidas: {STAGES}')
    bad = [c for c in cfg['classes'] if c not in cm.ALL_CLASSES]
    if bad:
        problems.append(f'classes desconhecidas: {bad}')
    for c in cfg['classes']:
        for key in ('bg_encoder_dir', 'nobg_encoder_dir'):
            if os.path.exists(cfg[key]):
                try:
                    cm.find_class_encoder(cfg[key], c)
                except FileNotFoundError as exc:
                    if key == 'bg_encoder_dir' or 'image' in cfg['stages']:
                        problems.append(str(exc))
    return problems


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default=None, help='JSON com chaves de CONFIG a sobrescrever.')
    p.add_argument('--classes', default=None, help='Sobrescreve CONFIG["classes"] (vírgula).')
    p.add_argument('--stages', default=None, help=f'Sobrescreve CONFIG["stages"] (vírgula). Válidas: {",".join(STAGES)}')
    p.add_argument('--force', default=None, help='Etapas a refazer (vírgula).')
    p.add_argument('--run_dir', default=None, help='Retoma/continua uma rodada existente.')
    p.add_argument('--html_only', action='store_true', help='Só gera o HTML da --run_dir.')
    p.add_argument('--dry_run', action='store_true', help='Mostra os comandos sem executar.')
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.classes:
        cfg['classes'] = [c.strip() for c in args.classes.split(',') if c.strip()]
    if args.stages:
        cfg['stages'] = [s.strip() for s in args.stages.split(',') if s.strip()]
    if args.force:
        cfg['force'] = [s.strip() for s in args.force.split(',') if s.strip()]
    apply_env(cfg)

    import report_html
    if args.html_only:
        if not args.run_dir:
            p.error('--html_only exige --run_dir')
        print(report_html.build(os.path.abspath(args.run_dir), cfg))
        return

    problems = validate(cfg)
    if problems:
        print('Configuração inválida:\n  - ' + '\n  - '.join(problems))
        sys.exit(2)

    if args.run_dir:
        run_dir = os.path.abspath(args.run_dir)
        if not os.path.isdir(run_dir):
            p.error(f'--run_dir não existe: {run_dir}')
        rc = os.path.join(run_dir, 'run_config.json')
        if os.path.isfile(rc):
            found = cm.read_json(rc)
            for k in ('val_fraction', 'split_seed'):
                if found.get(k) != cfg[k]:
                    p.error(f'{k} da rodada ({found.get(k)}) difere da configuração ({cfg[k]})')
            if os.path.normcase(os.path.abspath(found.get('orig_root', ''))) != os.path.normcase(os.path.abspath(cfg['orig_root'])):
                p.error(f'orig_root da rodada difere: {found.get("orig_root")}')
    else:
        run_dir = os.path.join(cfg['output_root'], f'run_{cm.timestamp()}')
        if not args.dry_run:
            os.makedirs(run_dir, exist_ok=False)

    if not args.dry_run:
        log_path = start_log_file(run_dir)
        print(f'Log completo: {log_path}')
    print(f'Rodada: {run_dir}\nClasses: {cfg["classes"]}\nEtapas: {cfg["stages"]}\n')
    pipe = Pipeline(cfg, run_dir, dry_run=args.dry_run)
    if not args.dry_run:
        cm.write_json(os.path.join(run_dir, 'pipeline_config.json'), cfg)
    started = time.time()
    results = {c: pipe.run_class(c) for c in cfg['classes']}
    if args.dry_run:
        print('\n[dry-run] comandos das etapas seguintes dependem de artefatos que ainda não existem '
              '(pesos, máscaras, audit); os caminhos acima podem mudar na execução real.')
        return
    print(f'\nPipeline: {sum(results.values())}/{len(results)} classes completas em {(time.time() - started) / 3600:.2f} h')
    html = report_html.build(run_dir, cfg)
    print(f'Relatório HTML: {html}')


if __name__ == '__main__':
    main()
