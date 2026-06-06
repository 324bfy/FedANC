import matplotlib.pyplot as plt

# 数据
centralized_dec = 0.0026276973
centralized_eve = 0.945448
federated_dec = 0.066681
federated_eve = 1.305495

labels = ['Centralized', 'Federated (FedANC)']

# 图1：Decoder MSE 对比（对数坐标）
plt.figure(figsize=(6,4))
bars = plt.bar(labels, [centralized_dec, federated_dec], color=['green', 'blue'])
plt.yscale('log')
plt.ylabel('Decoder MSE (log scale)')
plt.title('Comparison of Decryption Error')
for bar, val in zip(bars, [centralized_dec, federated_dec]):
    plt.text(bar.get_x() + bar.get_width()/2, val + 0.001, f'{val:.6f}', ha='center')
plt.savefig('comparison_decoder.png')
plt.close()

# 图2：Eve MSE 对比（线性坐标，越高越好）
plt.figure(figsize=(6,4))
bars = plt.bar(labels, [centralized_eve, federated_eve], color=['green', 'blue'])
plt.ylabel('Eve MSE (higher is better)')
plt.title('Comparison of Adversary Error')
for bar, val in zip(bars, [centralized_eve, federated_eve]):
    plt.text(bar.get_x() + bar.get_width()/2, val + 0.05, f'{val:.6f}', ha='center')
plt.savefig('comparison_eve.png')
plt.close()

print("Saved comparison_decoder.png and comparison_eve.png")