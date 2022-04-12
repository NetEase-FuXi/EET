save_file = open("../../resource/vocab.txt","w") 
dict_file = open('../../resource/dict.txt', 'r')
pre_file = open('./Special_Symbols.txt', 'r')

contents=pre_file.readlines()       #读取全部行
for content in contents:       #显示一行
    save_file.write(content)
save_file.write('\n')
pre_file.close()

contents=dict_file.readlines()       #读取全部行
for content in contents:       #显示一行
    save_file.write(content.split(' ')[0] + '\n')
dict_file.close()
save_file.close()
