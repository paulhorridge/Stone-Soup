class EvenNumbers:
    def __init__(self, max_number):
        self.number = 0
        self.max = max_number

    def __iter__(self):
        return self

    def __next__(self):
        if self.number > self.max:
            raise StopIteration
        self.number += 2
        return self.number - 2

even_iterator = EvenNumbers(10)
for num in even_iterator:
    print(num)

