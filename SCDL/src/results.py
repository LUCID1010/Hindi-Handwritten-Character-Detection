import matplotlib.pyplot as plt

plt.figure(figsize=(16,6))

# --




#------------------ Accuracy --------------------
plt.subplot(1,2,1)
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel("Number of epochs")
plt.ylabel("Model Accuracy")
plt.legend()

# -------------------- Loss ------------------------
plt.subplot(1,2,2)
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Number of epochs")
plt.ylabel("Model Data Loss")
plt.legend()

plt.show()
