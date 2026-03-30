#6.3的特征工程文件
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ========= 配置区 =========
input_npz = "features_mri.npz"      # 你的输入 .npz 文件
output_npz = "features_pca100.npz"  # 输出文件
feature_key = "X"                   # 特征矩阵在 .npz 里的 key
label_key = "y"                     # 标签 key（如果有，可选）
# ==========================

# 1) 读取 .npz
data = np.load(input_npz)
print("keys in npz:", data.files)

X = data[feature_key]  # 形状: (n_samples, n_features)
y = data[label_key] if label_key in data.files else None

print("X shape:", X.shape)

# 2) StandardScaler (按特征列)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3) PCA 降到 100 维
pca = PCA(n_components=100, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print("X_pca shape:", X_pca.shape)

# 4) 保存
if y is not None:
    np.savez(output_npz, X=X_pca, y=y)
else:
    np.savez(output_npz, X=X_pca)

print(f"Saved to {output_npz}")