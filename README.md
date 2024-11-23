# Мотивация

**Главная мотивация проекта** - закрепить теоритический материал по классическим моделям на практике. <br>
Для оценки качества реализаций проводится сравнение с аналогами из scikit-learn. <br>
Алгоритм считается успешно реализованными, если его качество по двум метрикам (для классификации - `accuracy`, `roc_auc`; для регрессии - `mse`, `r2`) приблизительно совпадает с качеством реализации из scikit-learn.

:warning: Вопросы дизайна и быстродействия алгоритмов были оставлены за скоупом проекта :warning:


# Конфигурация репозитория

Репозиторий содержит следующие файлы:
- `ModelEvaluation.py` - содержит модуль для оценки и сравнения моделей
- `Metrics.py` - содержит модуль с метриками
- `LinearModels.py` - содержит модуль с линейной и логистической регрессиией
- `EnsembleModels.py` - содержит модуль с случайным лесом и градиентным бустингом
- `DistanceBasedModels.py` - содержит модуль с методом ближайших соседей
- `%Testing.ipynb` - блокноты с результатами сравнения кастомных реализаций против scikit-learn
