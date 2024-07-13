import subprocess

def run_prepare_train_data():
    print("Running prepare_trainData.py...")
    subprocess.run(["python", "prepare_trainData.py"], check=True)

def run_pinn_train_model():
    print("Running pinn_trainModel.py...")
    subprocess.run(["python", "pinn_trainModel.py"], check=True)

def run_evaluate_train_model():
    print("Running evaluate_trainModel.py...")
    subprocess.run(["python", "evaluate_trainModel.py"], check=True)

if __name__ == "__main__":
    run_prepare_train_data()
    run_pinn_train_model()
    run_evaluate_train_model()

    print("All scripts executed successfully.")
