# Работа с кодом

Назначение проекта и команды: [README.md](README.md). Карта компонентов: [ARCHITECTURE.md](ARCHITECTURE.md). Научные определения: [docs/protocol.md](docs/protocol.md).

- Окружение: `uv sync --frozen`; проверка lock-файла: `uv lock --check`.
- Unit-проверки: `uv run python -m unittest discover -s tests`.
- Проверка состава MDP: `uv run python -m experiments plan --config experiments/mdp_factorial.json`.
- Запуск MDP: `uv run python -m experiments run --config experiments/mdp_factorial.json --workers 30`.
- Исходная статья: `uv run python -m experiments run --legacy --workers 30`.
- Legacy-подбор: `uv run python -m experiments tune --legacy --workers 30 --output results/tuning-study`.
- `src/astrorl` содержит только модели и общие компоненты; он не импортирует `experiments` и не читает файлы конфигурации.
- `experiments` читает конфиги, создаёт модели, управляет запуском и сохраняет результаты.
- Не добавляйте данные расчётов, отдельные графики, checkpoint, веса, кэши и архивы в Git. По явному запросу пользователя исторические итоговые PDF хранятся в `reports/` вместе с описанием источников; это исключение не распространяется на сырые результаты.
- Не меняйте бюджеты, seed, карты, метрики и правила остановки при структурных изменениях.
- SNN-архитектуры No Astro/AstroRL/Trace совпадают. RL не обновляет ALIF-ядро; предиктор обновляет его через локальное e-prop.
- MLP использует feedforward local eligibility/direct feedback (DFA). Trace — EMA familiarity bank.
- MLP сохранён только в legacy-протоколе; актуальные конфиги его не включают.
- В актуальных моделях память пишет actual RSNN rates и читает predicted RSNN rates. Real familiarity проверяется до записи достигнутого исхода; прогнозы ничего не записывают.
- Полный/локальный вход, число/перекрытие территорий,4/20I и tau задаются параметрами, без копий модели для каждого сочетания.
- `configs/experiments/` содержит ссылки на отдельные architecture/model/memory/motor/environment/protocol компоненты. Меняйте протокол только явно.
- `configs/conditions.json` и `.csv` должны оставаться согласованными.
- Длительные вычисления выполняются на вычислительном сервере с явно выбранным бюджетом ресурсов.
