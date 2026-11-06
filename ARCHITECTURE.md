# Архитектура

## Актуальные MDP-модели

```text
configs/experiments/<series>.json
  ├─ models/ → architecture + memory + motor + selected parameters
  ├─ environments/ + maps/
  └─ protocols/ → budget, seed, evaluation, statistics
                  ↓ experiments/composition.py
        experiments/models.py → объекты из src/astrorl
                  ↓
  modified.py + rollouts.py / checkpoint_rollout.py
                  ↓
  actual observation → RSNN → H(next actual RSNN activity)
                       ├─ actual activity: write memory
                       ├─ forecast: read-only familiarity query
                       └─ detached motor features
          familiarity →4/20 inhibitory LIF →4 Motor LIF
```

`src/astrorl/population.py` — одна модель для полного/локального входа, количества и перекрытия территорий, одинаковых/разных tau, single/pool. `adaptive.py` сохраняет отдельный механизм softmax-смешивания. `population_memory.py` содержит кальциевые/EMA-процессы и reliability; `dense_memory.py` — Whole/RFF. `pool_motor.py` объединяет4/20I общим параметром aggregation. `checkpoint_task.py` — общая dynamics reward-once задачи; `q_learning.py` — Q-table/Q-RNN.

Модели не читают JSON и не импортируют `experiments`. Factory получает уже разрешённые параметры. Код среды формирует RBF/context; астроцитарный learner получает признаки, а не координаты/optimal actions. В памяти только реальные RSNN-состояния. Real familiarity читается до записи исхода; observer сохраняет версию памяти для проверки причинного порядка.

`modified.py` задаёт репликации по картам/seed, проверяет manifest и неизменность весов evaluation, сохраняет метрики/навигацию. `modified_analysis.py` экспортирует training-AUC, графики, map-then-seed bootstrap и парные factorial-эффекты. Завершённые задачи кешируются, прерванная задача повторяется целиком. Конфиги latest MDP сохраняют исходные бюджеты250/300эпизодов,60финальных карт и seed72000..72002.

Подробнее: [модифицированные модели и конфиги](docs/modified_astro.md).

## Сохранённый протокол исходной статьи (`--legacy`)

```text
src/astrorl/
  config.py          ModelConfig и настройки prediction-обучения
  encoders.py        Общая фиксированная проекция для RSNN и памяти
  alif.py            ALIF-динамика и локальные eligibility traces
  backbone.py        Рекуррентное представление, калибровка и нормировка
  prediction.py      Предиктор следующего 24-мерного кода и один MSE
  motor.py           Четыре Motor LIF, inhibitory gating и локальные градиенты
  memory.py          Кальциевая память, EMA trace и familiarity
  mlp.py             MLP с локальным direct feedback
  plasticity.py      e-prop градиенты и ограничение шага ядра
  learner.py         Обучение предиктора и отделённой моторной головы
  environment.py     Переходы карты и shortest-path оценщик
  logging.py         Атомарные записи конфигураций, метрик и прогресса
experiments/
  __main__.py        Команды run / tune / plot / check
  configuration.py   Загрузка конфигураций и пути результатов
  runner.py          Организация обучения и анализа
  jobs.py            Задания, кеширование и пул процессов
  tuning.py          Кандидаты и ранжирование параметров
  engine.py          Эпизоды, evaluation, prediction probe и метрики
  analysis.py        Seed-level статистика, таблицы и рисунки
  checks.py          Научные инварианты и короткие интеграционные проверки
configs/             Входные параметры исследования
```

`astrorl` получает параметры через Python-объекты и не знает структуры файлов эксперимента. `experiments` создаёт `Learner` для каждой группы, модели, карты и seed; окружение и оценщик находятся вне модели. Карта и кратчайшие пути не передаются обучаемой policy.

ALIF-ядро обновляется только prediction-сигналом через локальные eligibility traces. RL обновляет отдельную моторную голову. Без RL моторные веса заморожены; память ON продолжает влиять на выбор действий через inhibitory gating. AstroRL и Trace используют одно интерфейсное правило: запись фактического состояния, чтение предсказанных переходов без записи.

Модели No Astro/AstroRL/Trace имеют общие ALIF, prediction head и четыре Motor LIF. Отличается только использование динамической памяти. MLP имеет собственные feedforward policy и prediction сети.

Evaluation сбрасывает динамическое состояние в каждом эпизоде и сохраняет обученные веса. Парные ON/OFF оценки используют одинаковые старты и последовательности случайных чисел. Единица независимой репликации в анализе — seed.
