import matplotlib.pyplot as plt


def plot_result(folder, all_history):

    for index, history in enumerate(all_history):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        epochs = range(len(acc))

        plt.figure(figsize=(15, 5))
        plt.plot(epochs, acc, 'b*-', label='Training accuracy')
        plt.plot(epochs, val_acc, 'r*-', label='Validation accuracy')
        plt.grid()
        plt.title('Training and validation accuracy')
        plt.ylabel("Accuracy")
        plt.xlabel("Epochs")
        plt.legend()
        plt.savefig(f'{folder}/fold-{index+1}-acc.png')
        plt.figure()
        plt.show()


        plt.figure(figsize=(15, 5))
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        plt.plot(epochs, loss, 'b*-', label='Training Loss')
        plt.plot(epochs, val_loss, 'r*-', label='Validation Loss')
        plt.grid()
        plt.title('Training and validation loss')
        plt.ylabel("Loss")
        plt.xlabel("Epochs")
        plt.legend()
        plt.savefig(f'{folder}/fold-{index+1}-loss.png')
        plt.figure()
        plt.show()
