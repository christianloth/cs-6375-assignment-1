
def stdout_output(*args):
    with open('log.txt', 'a') as log_file:
        if args:
            print(*args)
            log_file.write(' '.join(map(str, args)) + '\n')
        else:
            print()
            log_file.write('\n')


