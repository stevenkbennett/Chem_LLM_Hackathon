import json
from pathlib import Path

from langchain import OpenAI
from tabulate import tabulate
import sys




def score_task2_question(question, model_answer, student_answer, model):
    """Score a question in task 2."""
    # truncate answer to 110 words
    student_answer = " ".join(student_answer.split()[:110])
    query = f"""A student has been asked the following question: "{question}."

The correct answer for the question is: "{model_answer}"

The student has provided the following answer: "{student_answer}"
Rate the students answer using a number from 0 to 10. Do not explain your score."""
    mark = model(query).strip()
    try:
        mark = float(mark)
    except ValueError:
        mark = None
    return mark


def get_task2_score():
    """Get the total score for task 2 and summarise the score."""
    model = OpenAI(temperature=0.9)
    with open("task2_data.json", "r") as f:
        task2_data = json.load(f)

    if not Path("task2.txt").exists():
        return "No results submitted for task 2", None

    response = ""

    student_answers = Path("task2.txt").read_text().split("\n")
    if len(student_answers) > 10:
        response += "More than 10 answers provided. Ignoring any answers above line 10.\n\n"

    if len(student_answers) < 10:
        response += f"Only {len(student_answers)}/{len(task2_data)} answers were provided.\n\n"

    scores = []
    for i, (student_answer, question_data) in enumerate(zip(student_answers, task2_data)):
        score = score_task2_question(
            question_data["question"], question_data["model_answer"], student_answer, model=model
        )
        if score is None:
            response += "Error marking question {i}. Skipping...\n\n"
            score = 0
        scores.append(score)

    response += tabulate(
        zip(range(1, len(scores) + 1), scores),
        headers=["Question", "Score"],
        tablefmt="github",
    )
    final_score = sum(scores) / len(task2_data)
    return response, final_score


def get_task1_score():
    """Get the total score for task 1 and summarise the score."""
    sys.path.append(str(Path(__file__+"/../..").resolve()))
    from Task_1.eval import main as task1_eval
    if not Path("task_1_predictions.json").exists():
        return "No results submitted for task 1.", None
    task1_eval()
    with open('scores_task_1.txt', 'r') as f:
        response = f.read()
    # Get the integer in the line with total score in 
    score = int(response.split('\n')[-2].split(' ')[-1])
    return response, score


def get_total_score(task1_score, task2_score, task1_weight=0.5):
    if task1_score is None and task2_score is None:
        return "N/A"

    if task1_score is None:
        # This likely means the team is doing the afternoon session only.
        # The task2_score is weighted at 100%
        return f"{int(task2_score)}/90"

    if task2_score is None:
        # This is probably a morning session team that hasn't started on task 2.
        return f"{int(task1_score * task1_weight)}/90"

    return f"{int(task1_score * task1_weight + task2_score * (1 - task1_weight))}/90"


def get_response():
    """Get response and scores for all tasks"""
    try:
        task1_response, task1_score = get_task1_score()
    except Exception as e:
        print(e)
        task1_response = f"Error getting task 1 score."
        task1_score = None
    try:
        task2_response, task2_score = get_task2_score()
    except Exception as e:
        print(e)
        task2_response = f"Error getting task 2 score."
        task2_score = None
    # task2_response, task2_score = get_task2_score()
    total_score = get_total_score(task1_score, task2_score)

    response = "# Task 1 - Retrosynthesis\n\n"
    response += task1_response + "\n\n"
    response += "\n# Task 2 - Knowledge extraction\n\n"
    response += task2_response + "\n\n"
    response += "\n# Total score\n\n"
    response += total_score
    return response


def main():
    response = get_response()
    Path('score.txt').write_text(response)

if __name__ == "__main__":
    main()
