"""Execução e indexação de treinos (NF head e SEDifferNet) das etapas 6 e 7."""
import glob
import os
import subprocess
import sys
import time

import common as cm

TRAINER = os.path.join(cm.DIFFERNET_ROOT, 'scripts', 'train', 'pixel_train_from_pretrained.py')
ENCODER_RUNNER = os.path.join(cm.HERE, '_train_encoder_runner.py')


def default_se_checkpoint(class_name):
    return cm.find_class_encoder(cm.BG_ENCODER_DIR, class_name)


class ProgressAwareTee:
    """Separa a saída de um processo em linhas finais e atualizações de barra de progresso.

    Barras do tqdm se redesenham com ``\\r``. Aqui cada atualização sobrescreve a mesma linha
    do console, e o log (``log``) recebe só o estado final de cada linha, sem as atualizações
    intermediárias.
    """

    def __init__(self, log):
        self.log = log
        self.console = getattr(sys.stdout, 'console', sys.stdout)
        self.tty = hasattr(self.console, 'isatty') and self.console.isatty()
        self.pending = ''
        self.cr = False
        self.last_progress = ''
        self.shown = 0  # largura do texto de progresso visível no console

    def _show_progress(self, text):
        text = text.rstrip()
        if not text:
            return
        if self.tty:
            import shutil
            text = text[:max(20, shutil.get_terminal_size().columns - 1)]
        self.console.write('\r' + text + ' ' * max(0, self.shown - len(text)))
        self.console.flush()
        self.shown = len(text)

    def _finish_line(self, text):
        text = text.rstrip()
        if self.shown:
            self.console.write('\r' + ' ' * self.shown + '\r')
            self.shown = 0
        self.console.write(text + '\n')
        self.console.flush()
        extra = getattr(sys.stdout, 'file', None)  # log geral do run_all.py
        for f in (self.log, extra):
            if f is not None:
                f.write(text + '\n')
                f.flush()

    def feed(self, chunk):
        import re
        for part in re.split(r'(\r|\n)', chunk):
            if not part:
                continue
            if self.cr:  # '\r' anterior: '\r\n' é fim de linha (Windows); senão, redesenho de barra
                self.cr = False
                if part == '\n':
                    self._finish_line(self.pending)
                    self.pending = ''
                    continue
                if self.pending.strip():
                    self._show_progress(self.pending)
                    self.last_progress = self.pending
                self.pending = ''
            if part == '\n':
                self._finish_line(self.pending)
                self.pending = ''
            elif part == '\r':
                self.cr = True
            else:
                self.pending += part

    def close(self):
        if self.cr:
            self.feed('\n')
        if self.pending.strip():
            self._finish_line(self.pending)
        elif self.shown:
            # barra que terminou sem '\n' (tqdm com leave=False): grava o último estado
            self._finish_line(self.last_progress)
        self.pending = ''


def run_logged(cmd, log_path, cwd=cm.DIFFERNET_ROOT, env_extra=None):
    """Roda ``cmd`` espelhando a saída no console e em ``log_path``. Retorna (code, minutos).

    Barras de progresso aparecem numa linha só no console; no log fica só o estado final delas.
    """
    import codecs
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1', **(env_extra or {}))
    start = time.time()
    with open(log_path, 'w', encoding='utf-8') as log:
        log.write('$ ' + subprocess.list2cmdline(cmd) + '\n')
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
        tee = ProgressAwareTee(log)
        while True:
            data = proc.stdout.read1(65536)
            if not data:
                break
            tee.feed(decoder.decode(data))
        tee.feed(decoder.decode(b'', final=True))
        tee.close()
        code = proc.wait()
    return code, (time.time() - start) / 60


def latest_head_run(out_dir):
    runs = sorted(glob.glob(os.path.join(out_dir, 'run_*')))
    for run in reversed(runs):
        if os.path.isfile(os.path.join(run, 'best_models', 'best_pixel_auroc.pt')):
            return run
    return None


def head_is_complete(out_dir):
    run = latest_head_run(out_dir)
    return run is not None and os.path.isfile(os.path.join(run, 'final_results.csv'))


def trainer_command(class_name, dataset_root, se_checkpoint, out_dir, recipe, seed, split_seed,
                    val_fraction, epochs=None, extra=None, python=cm.PYTHON):
    cmd = [python, TRAINER, '--class_name', class_name, '--recipe', recipe, '--dataset', dataset_root,
           '--checkpoint', se_checkpoint, '--out_dir', out_dir, '--seed', str(seed),
           '--split_seed', str(split_seed), '--val_fraction', str(val_fraction)]
    if epochs:
        cmd += ['--epochs', str(epochs)]
    return cmd + list(extra or [])


def _class_relative(path, class_name):
    path = path.replace('\\', '/')
    prefix = class_name + '/'
    return path[len(prefix):] if path.startswith(prefix) else path


def check_split(head_run, audit_manifest_path):
    """Confere se o treino usou exatamente o inventário e o split do audit."""
    head = cm.read_json(os.path.join(head_run, 'split_manifest.json'))
    audit = cm.read_json(audit_manifest_path)
    head_paths = [_class_relative(s['path'], head['class_name']) for s in head['test_samples']]
    audit_paths = [s['path'] for s in audit['samples']]
    if head_paths != audit_paths:
        return False, 'inventário de teste difere'
    if [s['label'] for s in head['test_samples']] != [s['label'] for s in audit['samples']]:
        return False, 'rótulos diferem'
    val_audit = [i for i, s in enumerate(audit['samples']) if s['partition'] == 'val']
    if head['validation_indices'] != val_audit:
        return False, 'índices de validação diferem'
    return True, 'ok'


def update_index(class_dir, entry):
    path = os.path.join(class_dir, 'heads_index.json')
    index = cm.read_json(path) if os.path.isfile(path) else {'heads': []}
    index['heads'] = [h for h in index['heads'] if (h['arm'], h['variant']) != (entry['arm'], entry['variant'])]
    index['heads'].append(entry)
    index['updated_at'] = cm.timestamp()
    cm.write_json(path, index)
    return index


def load_index(class_dir):
    path = os.path.join(class_dir, 'heads_index.json')
    return cm.read_json(path) if os.path.isfile(path) else {'heads': []}


def train_head(class_name, class_dir, arm, variant, role, dataset_root, se_checkpoint, args, audit_manifest):
    out_dir = os.path.join(class_dir, f'arm_{arm}', variant, 'nf_head')
    cmd = trainer_command(class_name, dataset_root, se_checkpoint, out_dir, args.recipe, args.seed,
                          args.split_seed, args.val_fraction, args.epochs, args.trainer_args)
    if args.dry_run:
        print(f'[dry-run] {class_name} {arm}/{variant}: ' + subprocess.list2cmdline(cmd))
        return None
    if args.skip_existing and head_is_complete(out_dir):
        print(f'{class_name} {arm}/{variant}: head completa existente; pulei.')
        code, minutes = 0, 0.0
    else:
        code, minutes = run_logged(cmd, os.path.join(out_dir, f'train_{cm.timestamp()}.txt'))
    run = latest_head_run(out_dir)
    split_ok, split_msg = (check_split(run, audit_manifest) if run else (False, 'sem run'))
    entry = dict(arm=arm, variant=variant, role=role, dataset_root=dataset_root, se_checkpoint=se_checkpoint,
                 head_run=run, checkpoint=os.path.join(run, 'best_models', 'best_pixel_auroc.pt') if run else None,
                 status='SUCCESS' if code == 0 and run else 'FAILED', returncode=code, minutes=round(minutes, 2),
                 split_check=split_msg, recipe=args.recipe, seed=args.seed, split_seed=args.split_seed)
    update_index(class_dir, entry)
    print(f'{class_name} {arm}/{variant}: {entry["status"]} ({minutes:.1f} min), split={split_msg}')
    if not split_ok and run:
        print(f'   AVISO: split difere do audit ({split_msg}); a etapa 8 vai recusar esta head.')
    return entry
