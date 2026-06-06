import sys
from settings import update_config, defaults
from core.standalone import start_training      # 集中式训练入口
from core.evaluator import evaluate_model
from core.utils import str_to_bitlist, bitlist_to_str

# 联邦学习模块（仅在 fed 命令时导入）
try:
    from server import launch_federated
except ImportError:
    launch_federated = None

def main():
    args = sys.argv[1:]
    commands = [a for a in args if "=" not in a]
    update_config(args)   # 处理命令行参数覆盖

    if "help" in commands:
        print("Commands: train, eval, fed, encrypt, decrypt")
        return

    # 集中式训练
    if "train" in commands:
        start_training(model_name=defaults["active_model"])
        return

    # 联邦学习
    if "fed" in commands:
        if launch_federated is None:
            print("Federated module not available")
        else:
            launch_federated()
        return

    # 评估模型（需要预先训练好的模型）
    if "eval" in commands:
        evaluate_model()
        return

    # 加密演示
    if "encrypt" in commands:
        from core.evaluator import encrypt_message, generate_key, decode_tensor
        text = input("Text: ")
        key = generate_key()
        cipher = encrypt_message(text, key)
        if "decrypt" in commands:
            plain, guess = decrypt_message(cipher, key, with_eve=("eve" in commands))
            print("Decrypted:", decode_tensor(plain))
            if "eve" in commands:
                print("Eve guess:", decode_tensor(guess))
        else:
            print("Cipher:", cipher)
            print("Key:", key)
        return

    # 单独解密
    if "decrypt" in commands and "encrypt" not in commands:
        from core.evaluator import decrypt_message, decode_tensor
        cipher = input("Cipher: ")
        key = input("Key: ")
        plain, _ = decrypt_message(cipher, key)
        print("Decrypted:", decode_tensor(plain))
        return

if __name__ == "__main__":
    main()