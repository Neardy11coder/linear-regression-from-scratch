"""
Linear Regression from Scratch
Author: Hardik Kapil
Description:
Implements Linear Regression using Gradient Descent
without using any machine learning libraries.
"""

import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.m = 0  # slope
        self.b = 0  # intercept
        self.loss_history = []

    def fit(self, X, y):
        """
        Train the model using Gradient Descent
        """
        n = len(X)

        for _ in range(self.epochs):
            y_pred = self.m * X + self.b

            # Mean Squared Error
            loss = (1 / n) * np.sum((y - y_pred) ** 2)
            self.loss_history.append(loss)

            # Gradients
            dm = (-2 / n) * np.sum(X * (y - y_pred))
            db = (-2 / n) * np.sum(y - y_pred)

            # Update parameters
            self.m -= self.learning_rate * dm
            self.b -= self.learning_rate * db

    def predict(self, X):
        """
        Predict output values
        """
        return self.m * X + self.b


def main():
    # Sample Dataset
    X = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 6, 8, 10])

    # Train model
    model = LinearRegression(learning_rate=0.01, epochs=1000)
    model.fit(X, y)

    # Predictions
    y_pred = model.predict(X)

    # Output parameters
    print(f"Slope (m): {model.m}")
    print(f"Intercept (b): {model.b}")

    # Visualization
    plt.scatter(X, y, color="blue", label="Actual Data")
    plt.plot(X, y_pred, color="red", label="Regression Line")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title("Linear Regression from Scratch")
    plt.legend()
    plt.show()

    # Loss Curve
    plt.plot(model.loss_history)
    plt.xlabel("Epochs")
    plt.ylabel("Loss (MSE)")
    plt.title("Training Loss Curve")
    plt.show()


if __name__ == "__main__":
    main()
