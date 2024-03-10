# Тестовое задание для отборочных в VK
1. За основу модели была выбрана нейронная сеть с несколькими полносвязными слоями и функцией активации ReLU между ними.
2. На выходе бралась сигмойда для получения вероятностей предсказания класса
3. Модель оценивалась по метрике Normalized Discounted Cumulative Gain (ndcg_score в библиотека sklearn)
4. Модель обучалась на обучающей выборке, приложенной в файле train_df.csv и предсказывала данные на тестовой выборке test_df.csv
5. Для обучения было взято 50 эпох
### Точность модели на тестовой выборке на ndcg_score составила 0.9796123345747915 или 0.98 - что является отличным результатом
### Веса модели лежат в файле model_weights.pth
### Веса для скейлинга данных лежат в файле scaler.pkl
Сам код лежит в файле main.py
