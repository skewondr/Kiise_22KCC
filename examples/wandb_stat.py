import pandas as pd
import statistics
from IPython import embed
"""
train_valid_path = "/home/tako/yoonjin/pykt-toolkit/data/algebra2005/sub100_fold5/train_valid_sequences.csv"
test_path = "/home/tako/yoonjin/pykt-toolkit/data/algebra2005/sub100_fold5/test_sequences.csv"
# fold uid concepts responses selectmarks
train_valid_df = pd.read_csv(train_valid_path)
test_df = pd.read_csv(test_path)

tv_uids = set([])
tv_concepts = set([])
tv_interactions = 0
tv_total_wrongs = 0
tv_total_rights = 0
tv_seq_lens = []

for i, row in train_valid_df.iterrows():
    # print(row)
    row_uid = row["uid"]

    row_concepts = list(map(int, row["concepts"].split(",")))
    row_responses = list(map(int, row["responses"].split(",")))

    truncated_row_concepts = list(filter((-1).__ne__, row_concepts))
    truncated_row_responses = list(filter((-1).__ne__, row_responses))

    tv_uids.add(row["uid"])
    tv_uids.add(row_uid)

    tv_seq_lens.append(len(truncated_row_responses))

    tv_concepts.update(truncated_row_concepts)

    tv_total_wrongs += truncated_row_responses.count(0)

    tv_total_rights += truncated_row_responses.count(1)

print("Train&Valid")
print(f"uids: {len(tv_uids)}, concepts: {len(tv_concepts)}, interactions: {sum(tv_seq_lens)}")
print(f"Total wrongs : {tv_total_wrongs}, Total rights : {tv_total_rights}, Correct Ratio : {(tv_total_rights / (tv_total_rights + tv_total_wrongs)) * 100}, Wrong Ratio : {(tv_total_wrongs / (tv_total_rights + tv_total_wrongs)) * 100}")
print(f"Mean seq length: {statistics.mean(tv_seq_lens)}, Median seq length: {statistics.median(tv_seq_lens)}, Max seq length: {max(tv_seq_lens)}")

print()

test_uids = set([])
test_concepts = set([])
test_interactions = 0
test_total_wrongs = 0
test_total_rights = 0
test_seq_lens = []

for i, row in test_df.iterrows():
    # print(row)

    row_uid = row["uid"]

    row_concepts = list(map(int, row["concepts"].split(",")))
    row_responses = list(map(int, row["responses"].split(",")))

    truncated_row_concepts = list(filter((-1).__ne__, row_concepts))
    truncated_row_responses = list(filter((-1).__ne__, row_responses))

    test_uids.add(row_uid)

    test_seq_lens.append(len(truncated_row_responses))

    test_concepts.update(truncated_row_concepts)

    test_total_wrongs += truncated_row_responses.count(0)

    test_total_rights += truncated_row_responses.count(1)

print("Test")
print(f"uids: {len(test_uids)}, concepts: {len(test_concepts)}, interactions: {sum(test_seq_lens)}")
print(f"Total wrongs : {test_total_wrongs}, Total rights : {test_total_rights}, Correct Ratio : {(test_total_rights / (test_total_rights + test_total_wrongs)) * 100}, Wrong Ratio : {(test_total_wrongs / (test_total_rights + test_total_wrongs)) * 100}")
print(f"Mean seq length: {statistics.mean(test_seq_lens)}, Median seq length: {statistics.median(test_seq_lens)}, Max seq length: {max(test_seq_lens)}")

print()

total_ids = tv_uids.union(test_uids)
total_concepts = tv_concepts.union(test_concepts)
print(f"Total ids : {len(total_ids)}, Total concepts : {len(total_concepts)}, Total interactions : {sum(tv_seq_lens) + sum(test_seq_lens)}")
# print(f"Mean seq length: {statistics.mean(seq_lens)}, Median seq length: {statistics.median(seq_lens)}, Max seq length: {max(seq_lens)}")
"""
############################################################################################################################################################################
import pandas as pd
import statistics
from heapq import heappush, heappop

train_valid_path = "/home/tako/yoonjin/pykt-toolkit/data/assist2009/sub100_fold5/train_valid_sequences.csv"
test_path = "/home/tako/yoonjin/pykt-toolkit/data/assist2009/sub100_fold5/test_question_sequences.csv"
# fold uid concepts responses selectmarks
train_valid_df = pd.read_csv(train_valid_path)
test_df = pd.read_csv(test_path)

tv_uids = set([])
tv_concepts = set([])
tv_questions = set([])
tv_interactions = 0
tv_total_wrongs = 0
tv_total_rights = 0
tv_seq_lens = []

row_ratio = {}

for i, row in train_valid_df.iterrows():
    # print(row)
    # print(type(row))
    row_uid = row["uid"]

    row_questions = list(map(int, row["questions"].split(",")))
    row_concepts = list(map(int, row["concepts"].split(",")))
    row_responses = list(map(int, row["responses"].split(",")))

    truncated_row_questions = list(filter((-1).__ne__, row_questions))
    truncated_row_concepts = list(filter((-1).__ne__, row_concepts))
    truncated_row_responses = list(filter((-1).__ne__, row_responses))

    tv_uids.add(row["uid"])
    tv_uids.add(row_uid)

    tv_seq_lens.append(len(truncated_row_responses))

    tv_concepts.update(truncated_row_concepts)
    tv_questions.update(truncated_row_questions)

    tv_total_wrongs += truncated_row_responses.count(0)

    tv_total_rights += truncated_row_responses.count(1)
    row_ratio[row_uid] = (tv_total_rights / (tv_total_rights + tv_total_wrongs)) * 100
    # print(truncated_row_responses)

print("Train&Valid")
print(f"uids: {len(tv_uids)}, concepts: {len(tv_concepts)}, questions: {len(tv_questions)}, interactions: {sum(tv_seq_lens)}")
print(f"Total wrongs : {tv_total_wrongs}, Total rights : {tv_total_rights}, Correct Ratio : {(tv_total_rights / (tv_total_rights + tv_total_wrongs)) * 100}, Wrong Ratio : {(tv_total_wrongs / (tv_total_rights + tv_total_wrongs)) * 100}")
print(f"Mean seq length: {statistics.mean(tv_seq_lens)}, Median seq length: {statistics.median(tv_seq_lens)}, Max seq length: {max(tv_seq_lens)}")
sorted_row_ratio = sorted(row_ratio.items(), key=lambda x: x[1], reverse=True)
# for s in sorted_row_ratio[:100]:
#     print(s)
#     print()
print()

test_uids = set([])
test_concepts = set([])
test_questions = set([])
test_interactions = 0
test_total_wrongs = 0
test_total_rights = 0
test_seq_lens = []

for i, row in test_df.iterrows():
    # print(row)

    row_uid = row["uid"]

    row_questions = list(map(int, row["questions"].split(",")))
    row_concepts = list(map(int, row["concepts"].split(",")))
    row_responses = list(map(int, row["responses"].split(",")))

    truncated_row_questions = list(filter((-1).__ne__, row_questions))
    truncated_row_concepts = list(filter((-1).__ne__, row_concepts))
    truncated_row_responses = list(filter((-1).__ne__, row_responses))

    test_uids.add(row_uid)

    test_seq_lens.append(len(truncated_row_responses))

    test_concepts.update(truncated_row_concepts)
    test_questions.update(truncated_row_questions)

    test_total_wrongs += truncated_row_responses.count(0)

    test_total_rights += truncated_row_responses.count(1)

print("Test")
print(f"uids: {len(test_uids)}, concepts: {len(test_concepts)}, questions: {len(test_questions)}, interactions: {sum(test_seq_lens)}")
print(f"Total wrongs : {test_total_wrongs}, Total rights : {test_total_rights}, Correct Ratio : {(test_total_rights / (test_total_rights + test_total_wrongs)) * 100}, Wrong Ratio : {(test_total_wrongs / (test_total_rights + test_total_wrongs)) * 100}")
print(f"Mean seq length: {statistics.mean(test_seq_lens)}, Median seq length: {statistics.median(test_seq_lens)}, Max seq length: {max(test_seq_lens)}")

print()

total_ids = tv_uids.union(test_uids)
total_concepts = tv_concepts.union(test_concepts)
total_questions = tv_questions.union(test_questions)
print(f"Total ids : {len(total_ids)}, Total concepts : {len(total_concepts)}, Total questions : {len(total_questions)}, Total interactions : {sum(tv_seq_lens) + sum(test_seq_lens)}")
# print(f"Mean seq length: {statistics.mean(seq_lens)}, Median seq length: {statistics.median(seq_lens)}, Max seq length: {max(seq_lens)}")