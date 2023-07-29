epoch, loss, acc = 600, 1.2345, 0.87654321
print(f"{epoch:4d} {loss:5.2f} {acc:.2%}")


data = ['a', 'b', 'c']
for i, x in enumerate(data, start=1):
    print(i, x)

with open('./Deep_learning/test.txt', 'w') as f:
    print('zzz', file=f)  # manner 1
    f.write('zzz')  # manner 2

# zip 一起迭代
images = [1, 2, 3]
labels = ['a', 'b', 'c']
for img, label in zip(images, labels):
    print(img, label)

raw_data = ['1', '2', '3']
data = list(map(int, raw_data))
print(data)
data = list(map(lambda x: x*2, data))
print(data)

char_to_idx = {'a': 0, 'b': 1}
index_to_char = {v: k for k, v in char_to_idx.items()}
print(char_to_idx, index_to_char)

p1 = (1, 2)
p2 = (3, 4)
x0, y0, x1, y1 = *p1, *p2  # unpack
print(x0, y0, x1, y1, p1, p2)


def norm_square(x, y):
    return x ** 2 + y**2


vector = (3, 4)
print(norm_square(vector[0], vector[1]))
x, y = vector
print(norm_square(x, y))
print(norm_square(*vector))
# packing and unpacking "*" cannot be used alone
p1 = (1, 2)
x0, x1 = *p1,
*p1, = x0, x1

# Transpose
before = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]]
after = list(map(list, zip(*before)))
print(after)

# dictionary packing
bn_args = {
    'momentum': 2,
    'track_running_stats': False,
}


def test_func(momentum, track_running_stats, tse=0):
    print("ok")


test_func(**bn_args)
