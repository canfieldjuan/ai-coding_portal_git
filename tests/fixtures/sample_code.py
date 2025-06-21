# Sample Python code for testing
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

class Calculator:
    def add(self, a, b):
        return a + b
    
    def multiply(self, a, b):
        return a * b

if __name__ == "__main__":
    calc = Calculator()
    result = calc.add(5, 3)
    print(f"Result: {result}")
