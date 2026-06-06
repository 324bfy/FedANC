import torch
import os
from settings import defaults
from core.crypto_nets import EncryptorDecoder, Eavesdropper
from core.utils import str_to_bitlist, bitlist_to_str

def get_model_paths(model_type="basic"):
    folder = f"models/{model_type}"
    files = [f for f in os.listdir(folder) if f.endswith('.pth')]
    alice_path = bob_path = eve_path = None
    for f in files:
        if f.startswith("sender"):
            alice_path = os.path.join(folder, f)
        elif f.startswith("receiver"):
            bob_path = os.path.join(folder, f)
        elif f.startswith("eve"):
            eve_path = os.path.join(folder, f)
    return alice_path, bob_path, eve_path

def load_models(model_type="basic"):
    cfg = defaults[model_type]
    block_dim = cfg["block_dim"]
    alice = EncryptorDecoder(block_dim)
    bob = EncryptorDecoder(block_dim)
    eve = Eavesdropper(block_dim)

    # try to load from saved models
    a_path, b_path, e_path = get_model_paths(model_type)
    if a_path and b_path and e_path:
        alice.load_state_dict(torch.load(a_path))
        bob.load_state_dict(torch.load(b_path))
        eve.load_state_dict(torch.load(e_path))
    alice.eval()
    bob.eval()
    eve.eval()
    return alice, bob, eve

def encrypt_message(plain_str, key_str, model_type="basic"):
    alice, _, _ = load_models(model_type)
    plain_bits = str_to_bitlist(plain_str)
    key_bits = str_to_bitlist(key_str)[:defaults[model_type]["block_dim"]]
    # pad or truncate plain bits to multiple of block_dim
    block_dim = defaults[model_type]["block_dim"]
    if len(plain_bits) % block_dim != 0:
        plain_bits += [0] * (block_dim - len(plain_bits) % block_dim)
    cipher_blocks = []
    for i in range(0, len(plain_bits), block_dim):
        token = plain_bits[i:i+block_dim]
        token_t = torch.tensor(token, dtype=torch.float32).unsqueeze(0)
        key_t = torch.tensor(key_bits, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            c = alice(torch.cat([token_t, key_t], dim=1))
        cipher_blocks.append(c.squeeze().tolist())
    return cipher_blocks

def decrypt_message(cipher_blocks, key_str, with_eve=False, model_type="basic"):
    _, bob, eve = load_models(model_type)
    key_bits = str_to_bitlist(key_str)[:defaults[model_type]["block_dim"]]
    key_t = torch.tensor(key_bits, dtype=torch.float32).unsqueeze(0)
    plain_blocks = []
    guess_blocks = []
    for block in cipher_blocks:
        block_t = torch.tensor(block, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            p = bob(torch.cat([block_t, key_t], dim=1))
            g = eve(block_t) if with_eve else None
        plain_blocks.append(p.squeeze().tolist())
        if g is not None:
            guess_blocks.append(g.squeeze().tolist())
    # convert to string
    flat_bits = [int(round(b)) for block in plain_blocks for b in block]
    plain_str = bitlist_to_str(flat_bits)
    guess_str = None
    if with_eve:
        flat_guess = [int(round(b)) for block in guess_blocks for b in block]
        guess_str = bitlist_to_str(flat_guess)
    return plain_str, guess_str

def decode_tensor(tensor_list, to="str"):
    if to == "str":
        flat = [int(round(x)) for seq in tensor_list for x in seq]
        return bitlist_to_str(flat)
    return tensor_list

def generate_key(block_dim=16):
    import random, string
    return ''.join(random.choices(string.ascii_letters+string.digits, k=block_dim))

def evaluate_model():
    # placeholder – you can implement your own evaluation
    print("Evaluation not implemented in this simplified version.")