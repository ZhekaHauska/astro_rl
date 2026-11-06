# AstroRL

Модели и эксперименты для исследования навигации спайковых нейронных сетей с астроцитарной памятью. Сравниваются обучение предсказания без RL и совместное обучение предсказания с отделённой моторной RL-головой.

В ветке `modified_astro` доступны актуальные MDP-модели: локальные/полные астроцитарные территории,4/20 ингибирующих нейронов, адаптивное смешивание и разные способы familiarity. Память получает реальную активность RSNN100, запросы — прогноз следующей активности. RL не обновляет RSNN.

Архитектуры, память, моторы, среды и параметры хранятся отдельно; `configs/experiments/*.json` связывают их в конкретное исследование. [Описание модификаций и конфигов](docs/modified_astro.md).

## Установка

Linux x86_64, Python 3.10.12, [uv](https://docs.astral.sh/uv/).

```bash
uv sync --frozen
```

## Эксперименты

Проверить состав последнего MDP-эксперимента с несколькими астроцитами:

```bash
uv run python -m experiments plan --config experiments/mdp_factorial.json
```

Короткий запуск и графики:

```bash
uv run python -m experiments run --config experiments/smoke_factorial.json --workers 2 --output results/smoke-factorial
uv run python -m experiments plot --config experiments/smoke_factorial.json --output results/smoke-factorial
```

Полные конфиги: `mdp_factorial.json` —1440 обучений на60картах с3целями; `mdp_pool.json` и `mdp_adaptive.json` — по720 одноцелевых обучений. Длительные запуски выполняются на вычислительном сервере с выбранным бюджетом ресурсов.

[Документация изменений](docs/README.md) описывает модели, familiarity и запуск составных конфигов. В [каталоге отчётов](docs/experiment_reports.md) собраны пять исторических PDF: многокартовый MDP, tau × gate, адаптивные астроциты, ингибирующий пул и three-goal factorial. Исходные истории и веса не перенесены; новые обучающие серии для этих PDF не запускались.

### Исходный протокол статьи

Сохранён для явного воспроизведения: общий24-мерный входной код памяти/RSNN, MSE следующего кода и четыре карты. Он не используется конфигами модифицированных моделей.

Запуск прежних условий с параметрами из `configs/conditions.json`:

```bash
uv run python -m experiments run --legacy --workers 30
```

Полный подбор параметров и последующее обучение:

```bash
uv run python -m experiments tune --legacy --workers 30 --output results/tuning-study
```

Повторное построение графиков для завершённого запуска:

```bash
uv run python -m experiments plot --legacy --output results
```

Численные проверки локальных правил обучения и короткие проверки всех экспериментальных условий:

```bash
uv run python -m experiments check --legacy --output results/checks
```

Длительные расчёты выполняйте на вычислительном сервере в `tmux`. Число параллельных обучений по умолчанию — 30; каждый процесс использует один вычислительный поток. Запуск автоматически сохраняет прогресс и продолжает незавершённые задания. Журнал можно направить в файл: `... > run.log 2>&1`.

## Условия исходного исследования статьи

- Модели: MLP с локальным DFA-правилом, Random walk, No Astro, AstroRL, AstroRL (Trace).
- Карты: Open 13×13, Obstacles 7×7, Rooms 9×9, Branching 7×7.
- Две группы: без RL и с RL. По 10 seed на каждую комбинацию, всего 400 обучений.
- Обучение: 600 эпизодов, максимум 50 действий; фиксированная цель и случайный старт. Эпизод заканчивается при достижении цели.
- Evaluation: 100 новых эпизодов на seed. Для AstroRL и Trace память оценивается в режимах ON/OFF одного checkpoint; веса заморожены.
- Метрики: min–max normalized reward, оптимальные/фактические шаги, coverage, backtracking, collision/revisit rates, prediction-метрики и occupancy.

Результаты создаются в выбранном каталоге: таблицы метрик, evaluation, checkpoint и графики PDF/SVG/PNG. Артефакты расчётов исключены из Git.

## Структура

- `src/astrorl/` — ALIF, RSNN-activity модели, локальная/полная память, inhibitory pool, Whole/RFF и legacy-компоненты статьи.
- `experiments/` — CLI, runner, подбор, эпизоды, evaluation, научные проверки и анализ.
- `configs/` — архитектура, память, карты, условия и параметры исследования.
- `tests/` — проверки моделей, конфигураций и границ пакетов.
- `docs/` — протокол и определения метрик.

[Архитектура](ARCHITECTURE.md) · [Конфигурации](configs/README.md) · [Модифицированные модели](docs/modified_astro.md) · [Протокол статьи](docs/protocol.md)

## Проверки кода

```bash
uv run python -m unittest discover -s tests
```
