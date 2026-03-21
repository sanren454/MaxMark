import math
import sys
from typing import List, Tuple
import galois
import numpy as np
import scipy

def hamming_params(k: int) -> int:
    """Calculate the number of parity bits r for a data block of length k."""
    r = 0
    while 2**r < k + r + 1:
        r += 1
    return r

def encode_block_parity(data_bits: List[int]) -> List[int]:
    """
    Given data_bits, compute and return only parity bits for the block.
    根据原始数据，计算并返回冗余码
    """
    k = len(data_bits)
    r = hamming_params(k)
    n = k + r
    # Build full codeword in 1-indexed array for parity calc
    code = [None] * (n + 1)
    # Place data bits at non-parity positions
    j = 0
    for i in range(1, n + 1):
        if (i & (i - 1)) == 0:
            code[i] = 0  # placeholder
        else:
            code[i] = data_bits[j]
            j += 1
    # Calculate parity bits
    parity_bits: List[int] = []
    for i in range(r):
        pos = 2**i
        total = 0
        for bit_pos in range(1, n + 1):
            if bit_pos & pos and bit_pos != pos:
                total ^= code[bit_pos]
        parity_bits.append(total)
    return parity_bits

def encode_bitstring_separate(bitstring: str, block_size: int = 16) -> Tuple[str, str]:
    """
    Encode bitstring into data+parity format:
      - Returns (data_str, parity_str),
        where data_str is the original bits,
        parity_str is all parity bits concatenated for each block.
    将冗余码放在所有原始数据最后
    """
    bitstring = ''.join([c for c in bitstring if c in ('0', '1')])
    bits = [int(b) for b in bitstring]
    k = block_size
    r = hamming_params(k)
    parity_bits: List[int] = []
    # Pad bits to multiple of block_size
    if len(bits) % k != 0:
        bits += [0] * (k - (len(bits) % k))
    # Compute parity for each block
    for i in range(0, len(bits), k):
        block = bits[i:i+k]
        parity_bits.extend(encode_block_parity(block))
    data_str = ''.join(str(b) for b in bits)
    parity_str = ''.join(str(b) for b in parity_bits)
    return data_str, parity_str

def decode_bitstring_separate(data_str: str, parity_str: str, block_size: int = 16) -> Tuple[List[int], List[int]]:
    """
    Decode data_str using parity_str. Returns (corrected, syndromes)
    根据分块大小去分别对每个块进行纠错计算，返回[纠正后的数据，校正子]
    """
    data_bits = [int(b) for b in data_str]
    parity_bits = [int(b) for b in parity_str]
    k = block_size
    r = hamming_params(k)
    corrected: List[int] = []
    syndromes: List[int] = []
    num_blocks = len(data_bits) // k
    if len(parity_bits) != num_blocks * r:
        raise ValueError(f"Total parity length {len(parity_bits)} != blocks*r {num_blocks*r}")
    # Process each block
    for i in range(num_blocks):
        block = data_bits[i*k:(i+1)*k]
        block_parity = parity_bits[i*r:(i+1)*r]
        # Reconstruct full codeword for Hamming decode
        n = k + r
        code = [None] * (n + 1)
        # place parity and data
        j = 0
        for pos in range(1, n + 1):
            if (pos & (pos - 1)) == 0:  # parity position
                idx = int(math.log2(pos))
                # print(f'pos={pos},idx={idx}')
                # sys.exit()
                code[pos] = block_parity[idx]
            else:
                code[pos] = block[j]
                j += 1
        # Compute syndrome
        syndrome = 0
        for i_bit in range(r):
            pos = 2**i_bit
            total = 0
            for bpos in range(1, n + 1):
                if bpos & pos:
                    total ^= code[bpos]
            if total:
                syndrome += pos
        # Correct if needed
        if 1 <= syndrome <= n:
            code[syndrome] ^= 1
        # Extract corrected data bits
        for pos in range(1, n + 1):
            if (pos & (pos - 1)) != 0:
                corrected.append(code[pos])
        syndromes.append(syndrome)
    # corrected_str = ''.join(str(b) for b in corrected)
    return corrected, syndromes







def encode_bch_blocks(data: np.ndarray, bch, best_bch_paras) -> (np.ndarray, np.ndarray):
    """
    Split data into blocks, pad last block with zeros if needed,
    encode each block with BCH, and return (data_padded, parity_bits).
    Both are numpy arrays of dtype int (0/1).
    """
    # BCH Parameters 
    m,n,t,k,block_size,cost,Ptotal = best_bch_paras

    # Pad to multiple of block_size
    num_blocks = math.ceil(data.size / block_size)
    pad_len = num_blocks * block_size - data.size
    data_padded = np.concatenate([data, np.zeros(pad_len, dtype=int)])
    parity_list = []

    for i in range(num_blocks):
        block = data_padded[i*block_size:(i+1)*block_size]
        # Create full k-length message by zero-padding up to k bits
        message = np.concatenate([block, np.zeros(k - block_size, dtype=int)])
        # Encode to length-n codeword
        codeword = bch.encode(message)
        # Extract parity bits (last n-k bits)
        parity = codeword[k:]
        parity_list.append(parity)

    parity_bits = np.concatenate(parity_list)
    return data_padded, parity_bits

def decode_bch_blocks(data: np.ndarray, parity: np.ndarray, bch, best_bch_paras) -> (np.ndarray, np.ndarray):
    """
    Decode BCH-encoded data. Inputs:
      - data: 1D numpy array of padded data bits
      - parity: 1D numpy array of parity bits (concatenated per block)
      - bch: bch polynomial
      - best_bch_paras: bch parameters for the best acc case

    Returns:
      - corrected_data: numpy array of corrected bits (same length as data)
      - error_counts: numpy array of length num_blocks with number of corrected errors per block
    """

    # BCH Parameters
    m,n,t,k,block_size,cost,Ptotal = best_bch_paras


    num_blocks = data.size // block_size
    corrected = []
    error_counts = []

    for i in range(num_blocks):
        block = data[i*block_size:(i+1)*block_size]
        parity_block = parity[i*(n-k):(i+1)*(n-k)]
        # Reconstruct the full codeword: message of length k
        message = np.concatenate([block, np.zeros(k - block_size, dtype=int)])
        codeword = np.concatenate([message, parity_block])
        # Decode (returns corrected message of length k)
        decoded = bch.decode(codeword)
        # Re-encode to get corrected codeword
        corrected_codeword = bch.encode(decoded)
        # Count bit errors corrected (difference between codewords)
        errors = np.count_nonzero(codeword != corrected_codeword)
        error_counts.append(errors)
        # Extract corrected data bits (first block_size bits)
        corrected.extend(decoded[:block_size])

    return np.array(corrected, dtype=int), np.array(error_counts, dtype=int)



def smallest_m(b: int, t: int) -> int:
    """
    找到最小的 m，使得 2^m - 1 >= b + m*t。
    b: 块长（信息位数）
    t: 纠错能力（最大可纠正位数）
    返回值 m 为扩域阶数，用于 BCH 码设计。
    """
    m = 1
    while True:
        if (2**m - 1) >= (b + m * t):
            return m
        m += 1



# def get_bch_paras(secret_length,p_error,epsilon):
#     p = p_error
#     best_paras = None
#     distance = 48
#     for b in range(64,secret_length+1,distance):
#         N_blocks = math.ceil(secret_length/b)
#         for t in range(int(0.05*b),int(0.15*b)+1):
#             m = smallest_m(b,t)
#             n = 2**m -1
#             r = m*t
#             k = n - m * t
#             Pfail = sum(scipy.special.comb(b, i)*p**i*(1-p)**(b-i) for i in range(t+1,b+1))
#             Ptotal = 1 - (1-Pfail)**N_blocks
#             if Ptotal <= epsilon:
#                 cost = r/b
#                 if best_paras is None or cost < best_paras[5]:
#                     best_paras = (m,n,t,k,b,cost,Ptotal)
#     if best_paras is None:
#         best_paras = (7,127,10,57,48,None,None)
#     return best_paras

def get_bch_paras(secret_length, p_error, epsilon):
    p = p_error
    best_paras = None
    distance = 8
    for b in range(64, secret_length + 1, distance):
        N_blocks = math.ceil(secret_length / b)
        for t in range(int(0.05 * b), int(0.15 * b) + 1):
            m = smallest_m(b, t)
            n = 2 ** m - 1
            r = m * t
            k = n - m * t
            # Check for numerical stability in p**i and (1-p)**(b-i)
            Pfail = 0
            for i in range(t + 1, b + 1):
                try:
                    comb_value = scipy.special.comb(b, i, exact=True)
                    p_term = p ** i
                    q_term = (1 - p) ** (b - i)
                    
                    # Check if the terms are valid numbers
                    if not (math.isfinite(comb_value) and math.isfinite(p_term) and math.isfinite(q_term)):
                        continue  # Skip invalid values
                    
                    Pfail += comb_value * p_term * q_term

                except ValueError:  # Catch invalid values
                    continue  # Skip any invalid combination or power calculations

            Ptotal = 1 - (1 - Pfail) ** N_blocks
            
            if Ptotal <= epsilon:
                cost = r / b
                if best_paras is None or cost < best_paras[5]:
                    best_paras = (m, n, t, k, b, cost, Ptotal)

    if best_paras is None:
        best_paras = (7, 127, 10, 57, 48, None, None)

    return best_paras

    

### Reed–Solomon Code

def divisors(n: int) -> list[int]:
    """Return all positive divisors of n."""
    divs = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    return sorted(divs)

def get_n_rs_candidates(k_rs: int, m: int = 8) -> list[int]:
    """
    For GF(2^m), find all valid n_rs (<= 2^m-1) such that
    - n_rs divides (2^m - 1)
    - n_rs >= k_rs
    These n_rs support a primitive n_rs-th root of unity.
    
    对于阶码m，找到所有的满足条件的n_rs
    """
    q_minus_1 = 2**m - 1
    return [n for n in divisors(q_minus_1) if n >= k_rs]

def get_rs_paras(secret_length: int, p_error: float, m: int = 8, epsilon: float = 1e-4):
    """
    Optimize RS parameters for a given bit-length secret and bit error rate p_error.
    We search over block sizes (in bits, multiples of m) and RS(n_rs, k_rs) candidates.
    Returns (m, n_rs, k_rs, r_symbols, t_rs, block_size, cost, Ptotal).
    """
    p = p_error
    best = None
    ecc_backups = 1 # 冗余校验位备份次数
    # Symbol size m, so iterate block_size in multiples of m bits
    for block_size in range(m, secret_length + 1, m):       # 块长
        N_blocks = math.ceil(secret_length / block_size)    # 块数
        # number of data symbols per block
        k_rs = math.ceil(block_size / m)                    # 每块的符号数
        for n_rs in get_n_rs_candidates(k_rs, m):           # 找到最适合的冗余数r_rs,即最合适的n_rs
            r_symbols = n_rs - k_rs                         # 冗余符号数
            if r_symbols*m*N_blocks + secret_length > 16384:
                continue 
            t_rs = r_symbols // 2                           # 符号纠错能力（纠错几位符号）
            # symbol error probability: any bit in symbol wrong
            # 符号错误概率：符号中任意位错误
            p_symbol = 1 - (1 - p)**m
            # probability block has > t_rs symbol errors
            # 块中有超出纠错能力的错误数（符号数）的概率，只计算信息的符号，不计算冗余的符号，因为对所有的冗余符号都做了“多数备份+投票”
            Pfail = sum(
                math.comb(k_rs, i)
                * p_symbol**i
                * (1 - p_symbol)**(k_rs - i)
                for i in range(t_rs + 1, k_rs + 1)
            )
            Ptotal = 1 - (1 - Pfail)**N_blocks
            if Ptotal <= epsilon:
                r_bits = r_symbols * m
                cost = r_bits / block_size
                if best is None or cost < best[6]:
                    ecc_backups = min(6, (16384-secret_length)//(r_symbols * m * N_blocks))
                    best = (m, n_rs, k_rs, r_symbols, t_rs, block_size, cost, Ptotal, ecc_backups)
    if best is None:
        # Fallback: no feasible combination
        best = (m, None, None, None, None, None, None, None, None)
    return best


def encode_rs_bitstring(bit_data: np.ndarray, paras, GF, rs) -> list:
    m, n_rs, k_rs, r_symbols, t_rs, block_size, _, _,_ = paras
    # split into block_size-bit chunks
    blocks = [bit_data[i:i+block_size] for i in range(0, len(bit_data), block_size)]
    print(f'最后一块的长度={len(blocks[-1])}')
    parity_blocks = []
    for blk in blocks:
        # pad this block up to block size with 0
        pad_len = block_size - len(blk)
        blk_padded = np.pad(blk, (0, pad_len), constant_values=0)
        # pack into bytes
        byte_data = np.packbits(blk_padded)
        # cast to GF symbols
        msg_syms = GF(byte_data)
        # encode full RS(51,32), sperate data and parity
        code_syms = rs.encode(msg_syms)
        data_syms = code_syms[:k_rs]
        parity_syms = code_syms[k_rs:]
        parity_blocks.append(parity_syms)

    # print(f'最后一个校验块的符号数={len(parity_blocks[-1])}')
    return parity_blocks


def decode_rs_blocks(data_with_error: np.ndarray, parity_blocks: list, paras, GF, rs)-> np.ndarray:
    # 此处的data_with_error是原始数据的长度，但是包含错误，需要用parity去纠错，没有填充，需要在纠错前手动对最后一块填充0
    m, n_rs, k_rs, r_symbols, t_rs, block_size, _, _, ecc_backups = paras

    # Split data into blocks of original block_size
    blocks = [data_with_error[i:i+block_size] 
              for i in range(0, len(data_with_error), block_size)]
    decoded_bits = []

    for i, blk in enumerate(blocks):
        # Pad last block up to block_size with zeros
        pad_len = block_size - len(blk)
        blk_padded = np.pad(blk, (0, pad_len), constant_values=0)

        # Pack into bytes (symbols)
        byte_data = np.packbits(blk_padded)
        msg_syms_err = GF(byte_data)  # length = k_rs

        # Fetch parity symbols for this block
        parity_syms = parity_blocks[i]  # length = r_symbols

        # Reconstruct full codeword: data symbols + parity symbols
        code_syms_err = np.concatenate([msg_syms_err, parity_syms])

        # Decode via RS to get corrected codeword
        corrected_codeword = rs.decode(code_syms_err)

        # Extract corrected data symbols
        corrected_data_syms = corrected_codeword[:k_rs]

        # Unpack corrected symbols back to bits and truncate padding
        corrected_ints = corrected_data_syms.view(np.ndarray).astype(np.uint8)
        corrected_bits = np.unpackbits(corrected_ints)[:len(blk)]

        decoded_bits.append(corrected_bits)

    # Concatenate all decoded blocks and truncate to original length
    all_decoded = np.concatenate(decoded_bits)
    return all_decoded[:len(data_with_error)]

def parity_blocks_to_bitarray_8bit(parity_blocks):
    """
    把一个长度为 N_blocks 的 FieldArray 列表（每块 r_symbols 个 uint8 符号）
    打平成一维的比特流 (0/1) ndarray。
    """
    # 1) 拼成一个连续的 uint8 ndarray
    arr = np.concatenate([blk.view(np.ndarray).astype(np.uint8)
                          for blk in parity_blocks])
    # 2) unpackbits 直接把每个字节拆成 8 个比特
    #    结果 shape = (N_blocks * r_symbols * 8,)
    return np.unpackbits(arr)

def bitarray_to_parity_blocks_8bit(bit_array, paras, GF):
    """
    把一维比特流还原回 FieldArray 列表，每个元素 r_symbols 个符号。
    - bit_array: 1D uint8 array of 0/1, length = N_blocks * r_symbols * 8
    - paras: (m=8, n_rs, k_rs, r_symbols, ..., block_size, ...)
    - GF: galois.GF(2**8)
    """
    _, n_rs, k_rs, r_symbols, *_ = paras
    # 1) 先 packbits 回到 uint8 符号
    #    (会自动把长度填充到 8 的倍数，但我们保证输入长度是整除 8 的)
    sym_arr = np.packbits(bit_array)
    # 2) 计算块数
    N_blocks = len(sym_arr) // r_symbols
    # 3) reshape 成 (N_blocks, r_symbols)
    sym_matrix = sym_arr.reshape(N_blocks, r_symbols)
    # 4) 转回 FieldArray 列表
    return [GF(row) for row in sym_matrix]


from pyldpc import make_ldpc,encode, decode,get_message
def simulate_Pfail(n, d_v, d_c, p_error,trails=20):
    """
    Monte-Carlo estimation of the block failure probability for LDPC(n, d_v, d_c) under BSC(p_error).
    """
    H, G = make_ldpc(n, d_v, d_c, systematic=True, sparse=True)
    k = G.shape[1]
    snr_linear = (1 - p_error) / p_error
    snr_db = 10 * np.log10(snr_linear)

    fails = 0
    for i in range(trails):
        msg = np.random.randint(0, 2, size=k)
        y = encode(G, msg, snr=snr_db)
        decoded = decode(H, y, snr=snr_db, maxiter=50)
        if not np.array_equal(get_message(G, decoded), msg):
            fails += 1
    return fails /trails

# def get_ldpc_paras(secret_length: int, p_error: float, epsilon: float = 1e-4):
#     """
#     Optimize LDPC parameters for a given bit-length secret and bit error rate p_error.
#     We search over block sizes and LDPC(n, d_v, d_c) candidates.
#     Returns (n, k, d_v, d_c, block_size, redundancy_rate, Ptotal).
#     """
#     p = p_error
#     best = None

#     # Enumerate possible block sizes
#     for block_size in [128, 256, 512, 1024]:  # Possible block sizes (in bits)
#         if block_size > secret_length:
#             continue
#         N_blocks = math.ceil(secret_length / block_size)  # Number of blocks

#         # Enumerate possible degree distributions for LDPC codes
#         for d_v in [2, 3, 4]:  # Variable node degrees
#             for d_c in range(d_v + 1, d_v * 10):  # Check node degrees
            
#                 # Calculate the codeword length n and the rate R
#                 R = 1 - d_v / d_c
#                 n = int(block_size / R)  # This is where the issue arises. We need to adjust n correctly.
                
#                 if abs(n - round(n)) > 1e-9:
#                     continue
#                 n = int(round(n))

#                 # Ensure conventional LDPC constraints: d_c | n and d_c | (n * d_v)
#                 if n % d_c != 0 or (n * d_v) % d_c != 0:
#                     continue

#                 # Simulate block failure probability
#                 Pfail = simulate_Pfail(n, d_v, d_c, p_error)
#                 Ptotal = 1 - (1 - Pfail) ** N_blocks

#                 # Compute redundancy rate and other parameters
#                 r_bits = n - block_size  # Redundant bits per block
#                 redundancy_rate = r_bits / block_size
                
#                 # Check if total failure probability is within the acceptable range
#                 if Ptotal <= epsilon:
#                     if best is None or redundancy_rate < best[5]:
#                         best = (n, block_size, d_v, d_c, redundancy_rate, Ptotal)

#     if best is None:
#         # Fallback: no feasible combination
#         best = (None, None, None, None, None, None)
#     return best

# def generate_ldpc_matrices(block_size: int, d_v: int, d_c: int):
#     """
#     Generate the LDPC matrices (G and H) based on block size, variable node degree and check node degree.
#     """
#     R = 1 - d_v / d_c  # Code rate
#     n = int(block_size / R)  # Calculate the codeword length n
#     print(f'>>>> n ={n},block_size={block_size}')
#     # Generate the LDPC matrices using the pyldpc library
#     H, G = make_ldpc(n, d_v, d_c, systematic=True, sparse=True)

#     return G, H

# def encode_ldpc(secret, block_size, G, p_error):
#     """
#     Encode the given secret using LDPC, and return the list of parity blocks.
#     """
#     blocks = [secret[i:i + block_size] for i in range(0, len(secret), block_size)]
#     print(f'Last block length = {len(blocks[-1])}')
    
#     parity_blocks = []
#     for blk in blocks:
#         # Pad this block up to block size with 0
#         pad_len = block_size - len(blk)
#         blk_padded = np.pad(blk, (0, pad_len), constant_values=0)

#         snr_linear = (1 - p_error) / p_error
#         snr_db = 10 * np.log10(snr_linear)
        
#         # Perform LDPC encoding
#         encoded_blk = encode(G, blk_padded, snr=snr_db)
        
#         # Extract the parity bits (the last n-k bits)
#         parity = encoded_blk[block_size:]
#         parity_blocks.append(parity)

#     return parity_blocks

# def decode_ldpc(secret,parity,paras,H,p_error):
#     n, block_size, d_v, d_c, redundancy_rate, Ptotal = paras
#     data_blocks = [secret[i:i+block_size] for i in range(0, len(secret), block_size)]
#     parity_size = n-block_size
#     parity_blocks = [parity[i:i+parity_size] for i in range(0,len(parity),parity_size) ]

#     snr_linear = (1 - p_error) / p_error
#     snr_db = 10 * np.log10(snr_linear)

#     decoded_res= []
#     for i in range(0,len(data_blocks)):
#         code_word = np.concatenate(data_blocks[i],parity_blocks[i])
#         decoded = decode(H,code_word,snr=snr_db,maxiter=30)
#         decoded_res.append(decoded)
#     return np.concatenate(decoded_res)


def generate_ldpc_for_block_size(target_block_size: int, p_error: float = 0.1):
    """
    根据目标块大小生成兼容的 LDPC 码
    返回 G, H, actual_k, n, d_v, d_c
    """
    best_match = None
    min_diff = float('inf')
    
    # 尝试不同的度数组合
    for d_v in [2, 3]:
        for d_c in [4, 6, 8, 9, 10, 12]:
            if d_c <= d_v:
                continue
                
            # 计算估计的码长 n
            R = 1 - d_v/d_c
            estimated_n = int(target_block_size / R)
            
            # 调整 n 使其满足 LDPC 约束
            # 1. n 必须能被 d_c 整除
            # 2. (n * d_v) 必须能被 d_c 整除
            valid_n = None
            
            # 在估计值附近搜索有效 n
            for delta in range(-50, 51):
                n_test = estimated_n + delta
                if n_test <= target_block_size:
                    continue
                    
                if n_test % d_c == 0 and (n_test * d_v) % d_c == 0:
                    valid_n = n_test
                    break
            
            if valid_n is None:
                continue
                
            try:
                # 使用 pyldpc 生成矩阵
                H, G = make_ldpc(valid_n, d_v, d_c, systematic=True, sparse=True)
                actual_k = G.shape[1]  # pyldpc 返回的 G 矩阵的列数就是 k
                
                # 评估与目标 block_size 的差异
                diff = abs(actual_k - target_block_size)
                
                # 简单性能评估（可选，可根据需要扩展）
                # 这里我们只考虑大小匹配度
                if diff < min_diff:
                    min_diff = diff
                    best_match = (valid_n, actual_k, d_v, d_c, G, H, diff)
                    
            except Exception as e:
                continue
    
    # 回退方案
    if best_match is None:
        print(f"⚠️  Warning: No ideal parameters for block_size={target_block_size}. Using fallback parameters.")
        # 使用经典的 (1024, 512) 码作为回退
        d_v, d_c = 3, 6
        valid_n = 1024
        H, G = make_ldpc(valid_n, d_v, d_c, systematic=True, sparse=True)
        actual_k = G.shape[1]
        best_match = (valid_n, actual_k, d_v, d_c, G, H, abs(actual_k - target_block_size))
    
    n, actual_k, d_v, d_c, G, H, diff = best_match
    print(f"✅ LDPC parameters adjusted: target block_size={target_block_size}, actual k={actual_k} "
          f"(difference: {diff} bits), n={n}, d_v={d_v}, d_c={d_c}")
    
    return G, H, actual_k, n, d_v, d_c


def encode_with_pyldpc(secret_bits: np.ndarray, paras, target_block_size,p_error: float = 0.1):
    """
    使用 pyldpc 库进行编码，自动适应块大小
    """
    # 1. 生成适配的 LDPC 码
    G, H, actual_k, n, d_v, d_c = paras
    
    # 2. 处理块大小调整
    if actual_k != target_block_size:
        print(f"🔧 Block size adjusted from {target_block_size} to {actual_k} to satisfy LDPC constraints")
    
    # 3. 准备数据 - 确保长度是 actual_k 的整数倍
    num_blocks = (len(secret_bits) + actual_k - 1) // actual_k
    padded_length = num_blocks * actual_k
    
    if len(secret_bits) < padded_length:
        # 填充最后一块
        secret_bits = np.pad(secret_bits, (0, padded_length - len(secret_bits)), 'constant')
    elif len(secret_bits) > padded_length:
        # 截断多余部分
        secret_bits = secret_bits[:padded_length]
        print(f"⚠️  Warning: Secret bits truncated from {len(secret_bits)} to {padded_length} to fit block structure")
    
    # 4. 分块编码
    encoded_blocks = []
    
    # pyldpc 的 encode 函数需要 SNR 参数（即使对于 BSC 信道）
    # 对于二进制对称信道，我们可以使用一个合理的 SNR 值
    # SNR 与 p_error 的关系：SNR = 10*log10((1-p_error)/p_error)
    snr = 10 * math.log10((1 - p_error) / p_error) if p_error > 0 else 100
    
    for i in range(num_blocks):
        block = secret_bits[i*actual_k:(i+1)*actual_k]
        
        # 使用 pyldpc 的 encode 函数
        encoded_block = encode(G, block, snr)
        encoded_blocks.append(encoded_block)
    
    # 5. 合并结果
    encoded_bits = np.concatenate(encoded_blocks)
    return encoded_bits, actual_k, n, d_v, d_c, H

def decode_with_pyldpc(received_bits: np.ndarray, actual_k: int, n: int, d_v: int, d_c: int, H, G, p_error: float = 0.1):
    """
    使用 pyldpc 库进行解码
    """
    # 1. 验证输入长度
    if len(received_bits) % n != 0:
        raise ValueError(f"Received bits length {len(received_bits)} is not a multiple of codeword length {n}")
    
    num_blocks = len(received_bits) // n
    decoded_blocks = []
    
    # 2. 计算 SNR（与编码时相同）
    snr = 10 * math.log10((1 - p_error) / p_error) if p_error > 0 else 100
    
    # 3. 分块解码
    for i in range(num_blocks):
        block = received_bits[i*n:(i+1)*n]
        
        # 使用 pyldpc 的 decode 函数
        decoded_codeword = decode(H, block, snr)
        
        # 提取信息位
        decoded_message = get_message(G,decoded_codeword)
        decoded_blocks.append(decoded_message)
        print(f'>>> i ={i}, decoded_message len ={len(decoded_message)}')
    # 4. 合并结果
    decoded_bits = np.concatenate(decoded_blocks)
    return decoded_bits