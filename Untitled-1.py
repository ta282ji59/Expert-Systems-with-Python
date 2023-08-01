# %%
import os
import tkinter as tk
import tkinter.filedialog
import tkinter.messagebox
import math
import operator
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag

# %%
def get_text_file_path(message):
    # tkinterというGUIを用いて読み込みたいファイルを選択してもらう
    root = tk.Tk()
    root.withdraw()
    fTyp = [("","txt")]
    iDir = os.path.abspath(os.path.dirname('__file__'))
    tk.messagebox.showinfo('Text_Analysis_Tool',message)
    datafile = tk.filedialog.askopenfilename(filetypes = fTyp,initialdir = iDir)
    return(datafile)

# %%
def read_text_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        list = f.readlines()

    new_list = []

    for i in list:
        word = i.split()
        new_list.append(word)
    return(new_list)

ignore_list = [""," ", "  ", "   ", "    "] #list of items we want to ignore in our frequency calculations

def corpus_frequency(corpus_list, ignore = ignore_list, calc = 'freq', normed = False): #options for calc are 'freq' or 'range'
        freq_dict = {} #empty dictionary

        for tokenized in corpus_list: #iterate through the tokenized texts
            if calc == 'range': #if range was selected:
                tokenized = list(set(tokenized)) #this creates a list of types (unique words)

            for token in tokenized: #iterate through each word in the texts
                if token in ignore_list: #if token is in ignore list
                    continue #move on to next word
                if token not in freq_dict: #if the token isn't already in the dictionary:
                    freq_dict[token] = 1 #set the token as the key and the value as 1
                else: #if it is in the dictionary
                    freq_dict[token] += 1 #add one to the count

        ### Normalization:
        if normed == True and calc == 'freq':
            corp_size = sum(freq_dict.values()) #this sums all of the values in the dictionary
            for x in freq_dict:
                freq_dict[x] = freq_dict[x]/corp_size * 1000000 #norm per million words
        elif normed == True and calc == "range":
            corp_size = len(corpus_list) #number of documents in corpus
            for x in freq_dict:
                freq_dict[x] = freq_dict[x]/corp_size * 100 #create percentage (norm by 100)

        return(freq_dict)

# Assuming you already have a 'collocation_results' dictionary containing collocation scores
def high_val(stat_dict,hits = 20,hsort = True,output = False,filename = None, sep = "\t"):
        #first, create sorted list. Presumes that operator has been imported
        sorted_list = sorted(stat_dict.items(),key=operator.itemgetter(1),reverse = hsort)[:hits]

        if output == False and filename == None: #if we aren't writing a file or returning a list
            for x in sorted_list: #iterate through the output
                print(x[0] + "\t" + str(x[1])) #print the sorted list in a nice format

        elif filename is not None: #if a filename was provided
            outf = open(filename,"w") #create a blank file in the working directory using the filename
            for x in sorted_list: #iterate through list
                outf.write(x[0] + sep + str(x[1])+"\n") #write each line to a file using the separator
            outf.flush() #flush the file buffer
            outf.close() #close the file

        if output == True: #if output is true
            return(sorted_list) #return the sorted list

def read_words_from_file(file_path):
    with open(file_path, 'r') as file:
        words = file.read().split()
        # 単語を小文字に統一して返す
        return [word.lower() for word in words]
    
def find_common_words(file_paths):
    # 各ファイルの単語リストを読み込む
    word_lists = [read_words_from_file(file_path) for file_path in file_paths]

    # 共通する単語を見つける
    common_words = set(word_lists[0]).intersection(*word_lists[1:])

    return common_words


# %%
def find_least_similar_dataset(questioned_dataset, datasets):
    # Calculate the keyword frequencies for the questioned dataset
    questioned_freq_dict = corpus_frequency(questioned_dataset, ignore=ignore_list, calc='freq', normed=True)

    # Calculate the common top 20 keywords between questioned dataset and each dataset
    common_keywords_counts = {}
    for dataset_name, dataset in datasets.items():
        dataset_freq_dict = corpus_frequency(dataset, ignore=ignore_list, calc='freq', normed=True)
        common_keywords = set(questioned_freq_dict.keys()) & set(dataset_freq_dict.keys())
        common_keywords_counts[dataset_name] = len(common_keywords)

    # Find the dataset with the least common top 20 keywords
    least_similar_dataset = min(common_keywords_counts, key=common_keywords_counts.get)

    return least_similar_dataset

# # Example usage:
# datasets = {
#     "dataset1": corpus_list1,
#     "dataset2": corpus_list2,
# }

# # questioned_dataset_name = "questioned_dataset"
# questioned_dataset = corpus_list3 # ここに質問されたデータセットを指定してください

# # Find the least similar dataset and suggest to exclude it
# least_similar_dataset = find_least_similar_dataset(questioned_dataset, datasets)
# print("The least similar dataset is:", least_similar_dataset)

# %%
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('punkt', quiet=True)  # Download the 'punkt' resource (if you haven't done it before)

def pos_tag_english_text(text_list):
    pos_tags_list = []
    for text_tokens in text_list:
        pos_tags = nltk.pos_tag(text_tokens)
        pos_tags_list.append(pos_tags)
    return pos_tags_list

# %%
def find_word_context(text, target_word, window_size):
    # Tokenize the text into sentences and words
    sentences = sent_tokenize(text)
    words = [word_tokenize(sentence) for sentence in sentences]

    # Find the target word in the tokenized text
    target_indices = []
    for i, sentence in enumerate(words):
        for j, word in enumerate(sentence):
            if word.lower() == target_word.lower():
                target_indices.append((i, j))

    # Extract the context around the target word
    contexts = []
    for sentence_index, word_index in target_indices:
        start = max(0, word_index - window_size)
        end = min(len(words[sentence_index]), word_index + window_size + 1)
        context = words[sentence_index][start:end]
        contexts.append(" ".join(context))

    return contexts


# %%
def find_pos_pattern(text, target_word):
    # Tokenize the text into words
    words = word_tokenize(text)

    # Find the target word in the tokenized text
    target_indices = [i for i, word in enumerate(words) if word.lower() == target_word.lower()]

    # Extract the POS patterns following the target word
    pos_patterns = []
    for idx in target_indices:
        if idx < len(words) - 1:  # Ensure there's a word following the target word
            target_pos = pos_tag([words[idx]])[0][1]
            next_pos = pos_tag([words[idx + 1]])[0][1]
            pos_pattern = f"{target_word} + {next_pos}"
            pos_patterns.append(pos_pattern)

    return pos_patterns


# %%
def count_all_pos_patterns(dataset):
    # Join the sentences in the dataset into a single string
    text = ' '.join([' '.join(sentence) for sentence in dataset])

    # Tokenize the text into words
    words = word_tokenize(text)

    # Perform POS tagging on the words
    pos_tags = nltk.pos_tag(words)

    # Extract POS patterns for all words
    pos_patterns = []
    for word_idx in range(len(pos_tags) - 1):
        target_word = pos_tags[word_idx][0]
        target_pos = pos_tags[word_idx][1]
        next_pos = pos_tags[word_idx + 1][1]
        pos_pattern = f"{target_word} + {next_pos}"
        pos_patterns.append(pos_pattern)

    # Count identical POS patterns
    pos_pattern_counts = {}
    for pattern in pos_patterns:
        if pattern in pos_pattern_counts:
            pos_pattern_counts[pattern] += 1
        else:
            pos_pattern_counts[pattern] = 1

    return pos_pattern_counts

# %%
def find_least_pos_similar_dataset(questioned_dataset, datasets):
    # Calculate POS pattern counts for the questioned dataset
    questioned_pos_patterns = count_all_pos_patterns(questioned_dataset)

    # Calculate POS pattern counts for each dataset
    pos_pattern_counts_per_dataset = {}
    for dataset_name, dataset in datasets.items():
        pos_pattern_counts_per_dataset[dataset_name] = count_all_pos_patterns(dataset)

    # Calculate the difference in POS pattern counts between questioned dataset and each dataset
    pattern_differences = {}
    for dataset_name, pos_pattern_counts in pos_pattern_counts_per_dataset.items():
        difference = 0
        for pattern, count in questioned_pos_patterns.items():
            if pattern in pos_pattern_counts:
                difference += abs(count - pos_pattern_counts[pattern])
            else:
                difference += count
        pattern_differences[dataset_name] = difference

    # Find the dataset with the least similarity in POS patterns
    least_similar_dataset = min(pattern_differences, key=pattern_differences.get)

    return least_similar_dataset

# %%
def create_count_window():
    # 別のウィンドウを作成
    new_window = tk.Toplevel(window)
    new_window.title("count")

    # メッセージを表示するラベルを作成
    label = tk.Label(new_window, text="dataset1")
    label.pack(padx=20, pady=10)
    output_text1 = tk.Text(new_window, wrap=tk.WORD, height=10, width=40)
    output_text1.pack(padx=10, pady=5)
    
    label = tk.Label(new_window, text="dataset2")
    label.pack(padx=20, pady=10)
    output_text2 = tk.Text(new_window, wrap=tk.WORD, height=10, width=40)
    output_text2.pack(padx=10, pady=5)
    
    label = tk.Label(new_window, text="questioned dataset")
    label.pack(padx=20, pady=10)
    output_text3 = tk.Text(new_window, wrap=tk.WORD, height=10, width=40)
    output_text3.pack(padx=10, pady=5)
    

    high_list1 = high_val(freq_dict1, output = True)
    high_list2 = high_val(freq_dict2, output = True)
    high_list3 = high_val(freq_dict3, output = True)

    pos_pattern_counts1 = count_all_pos_patterns(corpus_list1)
    pos_pattern_counts2 = count_all_pos_patterns(corpus_list2)
    pos_pattern_counts3 = count_all_pos_patterns(corpus_list3)
    
    def format_pos_patterns(pos_pattern_counts):
        # テキストの整形
        formatted_text = ""
        for pattern, count in pos_pattern_counts.items():
            formatted_text += f"{pattern}: {count}\n"

        return formatted_text
    result_text1 = "—————top 20 of frequency————— \n"
    result_text1 += "\n".join([f"{index + 1}. {item[0]}" for index, item in enumerate(high_list1)])
    result_text1 += "\n\n———————POS pattern———————\n"
    result_text1 += format_pos_patterns(pos_pattern_counts1)
    
    result_text2 = "—————top 20 of frequency————— \n"
    result_text2 += "\n".join([f"{index + 1}. {item[0]}" for index, item in enumerate(high_list2)])
    result_text2 += "\n\n----------POS pattern----------\n"
    result_text2 += format_pos_patterns(pos_pattern_counts2)

    result_text3 = "—————top 20 of frequency————— \n"
    result_text3 += "\n".join([f"{index + 1}. {item[0]}" for index, item in enumerate(high_list3)])
    result_text3 += "\n\n----------POS pattern----------\n"
    result_text3 += format_pos_patterns(pos_pattern_counts3)
    
    # 実行結果をウィンドウに表示
    output_text1.delete(1.0, tk.END)  # 既存のテキストを削除
    output_text1.insert(tk.END, result_text1)
    output_text2.delete(1.0, tk.END)  # 既存のテキストを削除
    output_text2.insert(tk.END, result_text2)
    output_text3.delete(1.0, tk.END)  # 既存のテキストを削除
    output_text3.insert(tk.END, result_text3)  

    # leaset similar datasetメッセージを表示するラベルを作成
    questioned_dataset = corpus_list3 # ここに質問されたデータセットを指定してください

    # Find the least similar dataset and suggest to exclude it
    least_similar_dataset = find_least_similar_dataset(questioned_dataset, datasets)

    label = tk.Label(new_window, text="The least similar dataset is {} based on freqency".format(least_similar_dataset))
    label.pack(padx=20, pady=10)
    
def create_list_window():
    # 別のウィンドウを作成
    new_window = tk.Toplevel(window)
    new_window.title("list")

    # メッセージを表示するラベルを作成
    label = tk.Label(new_window, text="display the shared words/keywords of each dataset")
    label.pack(padx=20, pady=10)

    output_text = tk.Text(new_window, wrap=tk.WORD, height=10, width=40)
    output_text.pack(padx=10, pady=5)

    common_keywords = find_common_words([file_path1, file_path2, file_path3])
    shared_words_list = list(common_keywords)

    # Prepare the formatted result_text as a single string
    result_text = "\n".join(f"{index + 1}. {keyword}" for index, keyword in enumerate(shared_words_list))


    output_text.delete(1.0, tk.END)
    output_text.insert(tk.END, result_text)
    
def create_pos_window():
    # OKボタンをクリックしたときの処理
    def ok_button_clicked():
        target_word = entry_var.get()
        print(target_word)
        # エントリーウィジェットで入力されたテキストを取得
        if target_word == 'all':
            # 実行結果をウィンドウに表示
            pos_tags1 = pos_tag_english_text(corpus_list1)
            pos_tags2 = pos_tag_english_text(corpus_list2)
            pos_tags3 = pos_tag_english_text(corpus_list3)

            result_text1 = "\n".join([f"{item[0]} + {item[1]}" for pos_tags_sentence in pos_tags1 for item in pos_tags_sentence])
            result_text2 = "\n".join([f"{item[0]} + {item[1]}" for pos_tags_sentence in pos_tags2 for item in pos_tags_sentence])
            result_text3 = "\n".join([f"{item[0]} + {item[1]}" for pos_tags_sentence in pos_tags3 for item in pos_tags_sentence])
            
            output_text1.delete(1.0, tk.END)  # 既存のテキストを削除
            output_text1.insert(tk.END, result_text1)
            output_text2.delete(1.0, tk.END)  # 既存のテキストを削除
            output_text2.insert(tk.END, result_text2)
            output_text3.delete(1.0, tk.END)  # 既存のテキストを削除
            output_text3.insert(tk.END, result_text3)
        else:
            pos_patterns1 = find_pos_pattern(text1, target_word)
            pos_patterns2 = find_pos_pattern(text2, target_word)
            pos_patterns3 = find_pos_pattern(text3, target_word)
            
            contexts1 = find_word_context(text1, target_word, window_size=6)
            contexts2 = find_word_context(text2, target_word, window_size=6)
            contexts3 = find_word_context(text3, target_word, window_size=6)
            
            # pos_patternsを文字列に変換して改行して表示
            result_text1 = "\n".join(str(item) for item in pos_patterns1)
            result_text1 += "\n\n----------context----------\n"
            result_text2 = "\n".join(str(item) for item in pos_patterns2)
            result_text2 += "\n\n----------context----------\n"
            result_text3 = "\n".join(str(item) for item in pos_patterns3)
            result_text3 += "\n\n----------context----------\n"
            
            output_text1.delete(1.0, tk.END)  # 既存のテキストを削除
            output_text1.insert(tk.END, result_text1)
            for idx, context in enumerate(contexts1, start=1):
                output_text1.insert(tk.END, f"Context {idx}: {context}\n\n")
                
            output_text2.delete(1.0, tk.END)  # 既存のテキストを削除
            output_text2.insert(tk.END, result_text2)
            for idx, context in enumerate(contexts2, start=1):
                output_text2.insert(tk.END, f"Context {idx}: {context}\n\n")
                
            output_text3.delete(1.0, tk.END)  # 既存のテキストを削除
            output_text3.insert(tk.END, result_text3)
            for idx, context in enumerate(contexts3, start=1):
                output_text3.insert(tk.END, f"Context {idx}: {context}\n\n")

    # 別のウィンドウを作成
    new_window = tk.Toplevel(window)
    new_window.title("pos")

    # メッセージを表示するラベルを作成
    label = tk.Label(new_window, text="Please enter your target word:\n if input all, then you can show all pos.")
    label.pack(padx=20, pady=10)     
    # テキスト入力用のエントリーウィジェットを作成
    entry_var = tk.StringVar()  # 変数を作成
    entry = tk.Entry(new_window, textvariable=entry_var)
    entry.pack(padx=20, pady=10)

    # OKボタンを作成
    ok_button = tk.Button(new_window, text="OK", command=ok_button_clicked)
    ok_button.pack(pady=10)
    
    label = tk.Label(new_window, text="dataset1")
    label.pack(padx=20, pady=10)
    output_text1 = tk.Text(new_window, wrap=tk.WORD, height=10, width=40)
    output_text1.pack(padx=10, pady=5)
    
    label = tk.Label(new_window, text="dataset2")
    label.pack(padx=20, pady=10)
    output_text2 = tk.Text(new_window, wrap=tk.WORD, height=10, width=40)
    output_text2.pack(padx=10, pady=5)
    
    label = tk.Label(new_window, text="questioned dataset")
    label.pack(padx=20, pady=10)
    output_text3 = tk.Text(new_window, wrap=tk.WORD, height=10, width=40)
    output_text3.pack(padx=10, pady=5)

    # leaset similar datasetメッセージを表示するラベルを作成
    questioned_dataset = text3  # Replace with the questioned dataset
    # Find the least similar dataset and suggest to exclude it
    least_similar_dataset = find_least_pos_similar_dataset(questioned_dataset, datasets)

    label = tk.Label(new_window, text="The least similar dataset is {} based on pos pattern".format(least_similar_dataset))
    label.pack(padx=20, pady=10)
    

# ウィンドウを作成
window = tk.Tk()
window.title("Text_Analysis_Tool")
window.geometry('300x300')     # 表示画面サイズ（幅x高さ)

file_path3 = get_text_file_path('Select the .txt file that will be the \'Questioned dataset\'')
file_path1 = get_text_file_path('Select the .txt file that will be the \'First comparison target\'')
file_path2 = get_text_file_path('Select the .txt file that will be the \'Second comparison target\'')

text1 = open(file_path1, encoding="UTF-8").read()
text2 = open(file_path2, encoding="UTF-8").read()
text3 = open(file_path3, encoding="UTF-8").read()

corpus_list1 = read_text_file(file_path1)
corpus_list2 = read_text_file(file_path2)
corpus_list3 = read_text_file(file_path3)

freq_dict1 = corpus_frequency(corpus_list1, ignore=ignore_list, calc='freq', normed=True)
freq_dict2 = corpus_frequency(corpus_list2, ignore=ignore_list, calc='freq', normed=True)
freq_dict3 = corpus_frequency(corpus_list3, ignore=ignore_list, calc='freq', normed=True)

datasets = {
    "dataset1": corpus_list1,
    "dataset2": corpus_list2,
}

def button1_clicked():
    create_count_window()

def button2_clicked():
    create_list_window()

def button3_clicked():
    create_pos_window()
# ボタンを作成
button1 = tk.Button(window, text="count", command=button1_clicked)
button2 = tk.Button(window, text="list", command=button2_clicked)
button3 = tk.Button(window, text="pos", command=button3_clicked)

# ウィジェットを配置
button1.pack(pady=10)
button2.pack(pady=10)
button3.pack(pady=10)

# イベントループを開始
window.mainloop()



