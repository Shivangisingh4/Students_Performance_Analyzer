from analyzer import StudentAnalyzer

analyzer = StudentAnalyzer("data/students.csv")

analyzer.preprocess()
analyzer.performance_label()
analyzer.ml_score()

print(analyzer.get_top_students())
