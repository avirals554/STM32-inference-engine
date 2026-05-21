forward_map = {}
reverse_map = {}
n = 0


def mapping():
    global n
    with open("input.txt", "r") as i:
        while True:
            char = i.read(1)
            if not char:
                break
            if char not in forward_map:
                forward_map[char] = n
                n += 1
            else:
                continue


def reverse_mapping():
    global reverse_map
    reverse_map = {v: k for k, v in forward_map.items()}
