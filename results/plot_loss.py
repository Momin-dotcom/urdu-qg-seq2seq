import matplotlib.pyplot as plt

epochs = list(range(10))
train_losses = [5.4819, 4.6984, 4.1920, 3.8269, 3.5360, 3.3169, 3.1421, 2.9956, 2.8804, 2.7525]
valid_losses = [5.1906, 4.7382, 4.6074, 4.5638, 4.5794, 4.6022, 4.6606, 4.7210, 4.7841, 4.8290]

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_losses, marker='o', label='Train Loss')
plt.plot(epochs, valid_losses, marker='o', label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss per Epoch')
plt.legend()
plt.grid(True)
plt.savefig('results/figures/loss_curve.png', dpi=150, bbox_inches='tight')
plt.show()