import sys

# 0: Nothing, 1: Warning, 2: Debug (light), 3: Debug (moderate), 4: Debug (detailed), 5: Debug (very detailed)
verbose_level = 3

def set_verbose_level(lv):
    global verbose_level
    verbose_level = lv

def messenger(head, *messages):
    is_head = True
    for msg in messages:
        if is_head:
            print(head, msg, end = '')
        else:
            print('', msg, end = '')
        is_head = isinstance(msg, str) and msg.endswith('\n')
    print('')

def error(*messages):
    messenger('[Error]', *messages)
    sys.exit(1)

def warning(*messages):
    global verbose_level
    if verbose_level >= 1:
        messenger('[Warning]', *messages)

def verbose(threshold, *messages):
    global verbose_level
    assert(threshold >= 2)
    if verbose_level >= threshold:
        head = '[Verbose]' + (' ' * (threshold - 2))
        messenger(head, *messages)
