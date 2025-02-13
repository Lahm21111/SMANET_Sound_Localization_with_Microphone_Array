import pickle

# 读取 pickle 文件
with open(r'C:\Users\grizi\Downloads\sslr\sslr\sslr\lsp_train_301\gt_file\ssl-data_2017-05-07-17-25-56_5.gt.pkl', 'rb') as f:
    data = pickle.load(f)


print(data)
