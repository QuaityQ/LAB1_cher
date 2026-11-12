"""Заготовки задач на базовый Python."""

from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput


def count_vowels(data: TextInput) -> int:
    text = data.value.lower()
    count = 0
    for char in text:
        if char in 'aeiou':
            count += 1
    return count


def has_unique_characters(data: TextInput) -> bool:
    text = data.value
    seen = set()
    for char in text:
        if char in seen:
            return False
        seen.add(char)
    return True


def count_one_bits(data: PositiveIntegerInput) -> int:
    number = data.value
    count = 0
    while number > 0:
        count += number % 2
        number = number // 2
    return count


def multiplicative_persistence(data: PositiveIntegerInput) -> int:
    number = data.value
    steps = 0
    while number >= 10:
        product = 1
        for digit in str(number):
            product *= int(digit)
        number = product
        steps += 1
    return steps


def mse(data: VectorPairInput) -> float:
    predicted, expected = data.predicted, data.expected
    total = 0
    for i in range(len(predicted)):
        total += (predicted[i] - expected[i]) ** 2
    return total / len(predicted)



def prime_factorization(data: PositiveIntegerInput) -> str:
    number = data.value
    result = ""
    divisor = 2
    while number > 1:
        power = 0
        while number % divisor == 0:
            number = number // divisor
            power += 1
        if power == 1:
            result += "(" + str(divisor) + ")"
        elif power > 1:
            result += "(" + str(divisor) + "**" + str(power) + ")"
        divisor += 1
    return result



def pyramid(data: PositiveIntegerInput) -> int | str:
    cube_count = data.value
    total = 0
    k = 0
    while total < cube_count:
        k += 1
        total += k * k
    if total == cube_count:
        return k
    return "It is impossible"



def is_balanced_number(data: PositiveIntegerInput) -> bool:
    digits = str(data.value)
    half = (len(digits) - 1) // 2
    left_sum = 0
    for digit in digits[:half]:
        left_sum += int(digit)
    right_sum = 0
    for digit in digits[len(digits) - half:]:
        right_sum += int(digit)
    return left_sum == right_sum


def _close(a, b):
    return abs(a - b) < 1e-9


# тесты

def test_count_vowels():
    assert count_vowels(TextInput("hello")) == 2
    assert count_vowels(TextInput("AEIOU")) == 5  # регистр не важен
    assert count_vowels(TextInput("Programming")) == 3
    assert count_vowels(TextInput("xyz")) == 0  # гласных нет
    assert count_vowels(TextInput("")) == 0  # пустая строка


def test_has_unique_characters():
    assert has_unique_characters(TextInput("abc")) is True
    assert has_unique_characters(TextInput("abakr")) is False
    assert has_unique_characters(TextInput("aba")) is False
    assert has_unique_characters(TextInput("a")) is True
    assert has_unique_characters(TextInput("")) is True
    assert has_unique_characters(TextInput("aA")) is True  # a и A разные символы


def test_count_one_bits():
    assert count_one_bits(PositiveIntegerInput(1)) == 1
    assert count_one_bits(PositiveIntegerInput(2)) == 1
    assert count_one_bits(PositiveIntegerInput(7)) == 3  # 111
    assert count_one_bits(PositiveIntegerInput(13)) == 3  # 1101
    assert count_one_bits(PositiveIntegerInput(255)) == 8  # 11111111
    assert count_one_bits(PositiveIntegerInput(1024)) == 1  # степень двойки


def test_multiplicative_persistence():
    assert multiplicative_persistence(PositiveIntegerInput(39)) == 3
    assert multiplicative_persistence(PositiveIntegerInput(4)) == 0  # уже однозначное
    assert multiplicative_persistence(PositiveIntegerInput(999)) == 4
    assert multiplicative_persistence(PositiveIntegerInput(10)) == 1  # 1*0 = 0
    assert multiplicative_persistence(PositiveIntegerInput(25)) == 2  # 10, 0
    assert multiplicative_persistence(PositiveIntegerInput(77)) == 4  # 49, 36, 18, 8


def test_mse():
    assert _close(mse(VectorPairInput([1, 2, 3], [1, 4, 6])), 13 / 3)
    assert _close(mse(VectorPairInput([5, 3, 4], [5, 5, 1])), 13 / 3)
    assert mse(VectorPairInput([1, 2, 3], [1, 2, 3])) == 0  # одинаковые векторы
    assert mse(VectorPairInput([5], [2])) == 9  # один элемент
    assert _close(mse(VectorPairInput([-1, -2], [1, 2])), 10)  # отрицательные


def test_prime_factorization():
    assert prime_factorization(PositiveIntegerInput(86240)) == "(2**5)(5)(7**2)(11)"
    assert prime_factorization(PositiveIntegerInput(12)) == "(2**2)(3)"
    assert prime_factorization(PositiveIntegerInput(13)) == "(13)"  # простое число
    assert prime_factorization(PositiveIntegerInput(2)) == "(2)"
    assert prime_factorization(PositiveIntegerInput(8)) == "(2**3)"  # степень двойки
    assert prime_factorization(PositiveIntegerInput(49)) == "(7**2)"


def test_pyramid():
    assert pyramid(PositiveIntegerInput(1)) == 1
    assert pyramid(PositiveIntegerInput(5)) == 2
    assert pyramid(PositiveIntegerInput(14)) == 3
    assert pyramid(PositiveIntegerInput(30)) == 4
    assert pyramid(PositiveIntegerInput(55)) == 5
    for n in (2, 3, 4, 6, 29):
        assert pyramid(PositiveIntegerInput(n)) == "It is impossible"


def test_is_balanced_number():
    assert is_balanced_number(PositiveIntegerInput(1234006)) is True
    assert is_balanced_number(PositiveIntegerInput(123456)) is False
    assert is_balanced_number(PositiveIntegerInput(7)) is True  # одна цифра
    assert is_balanced_number(PositiveIntegerInput(11)) is True  # две цифры
    assert is_balanced_number(PositiveIntegerInput(1221)) is True  # чётная длина
    assert is_balanced_number(PositiveIntegerInput(1232)) is False
    assert is_balanced_number(PositiveIntegerInput(12321)) is True  # нечётная длина























