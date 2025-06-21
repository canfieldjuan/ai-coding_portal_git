// Sample JavaScript code for testing
function fibonacci(n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

class Calculator {
    add(a, b) {
        return a + b;
    }
    
    multiply(a, b) {
        return a * b;
    }
}

const calc = new Calculator();
const result = calc.add(5, 3);
console.log(`Result: ${result}`);
