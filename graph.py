import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Create a simple probability density function (PDF) with alpha = 0.5
alpha = 0.1
x1 = np.linspace(5, 5, 100)
pdf1 = norm.pdf(x1, loc=0, scale=1) * alpha * 2
plt.plot(x1, pdf1, 'g--', label=f"Simple PDF (alpha={alpha})")

# Create a multimodal PDF
x2 = np.linspace(-10, 10, 200)
pdf2 = (norm.pdf(x2, loc=-3, scale=1) + norm.pdf(x2, loc=3, scale=2)) / 2
plt.plot(x2, pdf2, 'k', label="True distribution")

# Create a single mode PDF that covers only the multimodal PDF
pdf3 = pdf2.copy()
pdf3[60:140] = norm.pdf(x2[60:140], loc=0, scale=1) * 0.5
plt.plot(x2, pdf3, 'r--', label="Final probability")

# Set the title and legend
plt.title("Probability density functions")
plt.legend()

# Display the plot
plt.show()
