try:
    with open('output2.txt', 'r', encoding='utf-16') as f:
        print(f.read())
except Exception as e:
    print(e)
    try:
        with open('output2.txt', 'r', encoding='utf-8') as f:
            print(f.read())
    except:
        pass
