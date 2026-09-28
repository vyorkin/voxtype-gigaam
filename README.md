# voxtype-gigaam

Локальный распознаватель речи, подключённый к voxtype как OpenAI-совместимый эндпоинт.
voxtype остаётся фронтендом: горячая клавиша, OSD, focus-guard шим для `wtype`, waybar-статус —
всё как было, меняется только движок распознавания.

Три движка, выбор автоматический:

* **GigaAM v3 e2e RNN-T** (int8, 230 МБ) — русский, с пунктуацией и заглавными. Основной.
* **whisper large-v3-turbo** (GGML, Vulkan) — мультиязычный. Держит английские слова внутри
  русской речи латиницей и пишет их правильно. Обслуживается долгоживущим `whisper-server`.
* **Parakeet TDT 0.6B v3** (int8, ~600 МБ на диске) — мультиязычный, запасной вариант на
  случай, если whisper-сервер недоступен. Грузится лениво.

Русская модель запускается всегда. Дальше один предохранитель:

1. Если её вывод не содержит кириллицы (`let's stry something els` — это GigaAM на английской
   речи) или в нём есть латинские слова (`VoxTipe`, `Rast и Tipecript` — код-свитчинг),
   фраза перепроходит через whisper.

Уверенность между моделями **не сравнивается**: средний логвероятность токена у разных
моделей на одной и той же шкале не лежит (Parakeet на том же аудио был выше GigaAM на ~0.09,
из-за чего выигрывал 107 раз из 126, в 90 случаях портя русскую фразу). Решение принимает
только письменность вывода GigaAM.

Чистая русская речь предохранителя не замечает: второй декод не запускается, всё как раньше.

## Зачем

| | GigaAM (CPU) | whisper `large-v3-turbo` (Vulkan, GTX 1070 Ti) |
|---|---|---|
| Роль | чистая русская речь (быстро, с пунктуацией) | английская и смешанная речь (точные английские слова) |
| Модели на диске | 230 МБ | 1.6 ГБ |
| Русская фраза 6.5 с | 0.35 с декода | — (не используется) |
| Смешанная/английская фраза 8.4 с | — | ~1.8 с декода (модель загружена) |
| Пунктуация и заглавные по-русски | из коробки | слабее |
| RAM после прогрева | ~520 МБ | ~1.6 ГБ |

В гибриде фраза декодируется сначала GigaAM; на английском/код-свитчинге добавляется один
декод whisper. Чистая русская диктовка платит только ~0.05× от длительности фразы
(13 с речи — 0.65 с).

## Состав

| Путь | Что это |
|---|---|
| `server.py` | HTTP-сервер: `POST /v1/audio/transcriptions` (multipart, поле `file`), ответ `{"text": "..."}`. Плюс `GET /health`. |
| `.venv/` | Python 3.14 + `onnx-asr[cpu,hub]` (тянет `onnxruntime`, `numpy`). |
| `samples/ru_sample.wav`, `samples/en_sample.wav` | Тестовые фразы (TTS) для `--selftest` и `bench-stt`. |
| `bench-stt` | Сравнение бэкендов на обоих сэмплах (или на переданных WAV). |
| `analyze-dumps` | Прогон всех моделей по записанному сервером аудио (см. «Диагностика точности»). |
| `replacements.tsv` | Словарь замен для voxtype: секции, 139 активных правил (см. «Словарь замен»). |
| `~/.local/bin/voxtype-replacements` | Применение/откат словаря в `config.toml`. |
| `~/.config/systemd/user/voxtype-gigaam.socket` | Сокет-активация на `127.0.0.1:9017`. |
| `~/.config/systemd/user/voxtype-gigaam.service` | Сам сервер (запускается по первому запросу). |
| `~/.local/share/whisper.cpp` | Сборка whisper.cpp с Vulkan. Используется бинарник `build/bin/whisper-server` (второе мнение для английского). |
| `~/.config/systemd/user/whisper-server.service` | Долгоживущий `whisper-server` с large-v3-turbo на `127.0.0.1:9018`. |
| `~/.local/bin/voxtype-stt` | Переключатель профилей. |
| `~/.local/bin/voxtype-target` | Запоминает окно, в котором началась диктовка (focus guard). |
| `~/.local/share/voxtype-shim/wtype` | Шим `wtype`: focus guard и печать по словам (см. «Печать в Electron-приложениях»). |
| `~/.config/systemd/user/voxtype.service.d/path.conf` | Ставит шим впереди `/usr/bin/wtype` в PATH демона. |
| `~/.local/state/voxtype-stt/` | Состояние переключателя: активный профиль и сохранённые значения whisper. |
| `~/.cache/huggingface/hub/models--istupakov--gigaam-v3-onnx` | Веса русской модели. Загрузка одноразовая, дальше сервис работает офлайн (`HF_HUB_OFFLINE=1`). |
| `~/.cache/huggingface/hub/models--istupakov--parakeet-tdt-0.6b-v3-onnx` | Веса английской модели, тот же кэш. |

## Установка

Файлы в репозитории: `server.py`, `analyze-dumps`, `bench-stt`, `replacements.tsv`,
`samples/`, `bin/voxtype-{stt,replacements,target}`, `shim/wtype`,
`systemd/voxtype-gigaam.{socket,service}`, `systemd/whisper-server.service`,
`systemd/voxtype.service.d/path.conf`.

```sh
git clone https://github.com/vyorkin/voxtype-gigaam ~/.local/share/voxtype-gigaam
cd ~/.local/share/voxtype-gigaam
python -m venv .venv && .venv/bin/pip install 'onnx-asr[cpu,hub]' numpy
install -Dm755 bin/* ~/.local/bin/
install -Dm755 shim/wtype ~/.local/share/voxtype-shim/wtype
install -Dm644 systemd/voxtype-gigaam.service systemd/voxtype-gigaam.socket \
    systemd/whisper-server.service ~/.config/systemd/user/
install -Dm644 systemd/voxtype.service.d/path.conf \
    ~/.config/systemd/user/voxtype.service.d/path.conf
systemctl --user daemon-reload
# в ~/.config/voxtype/config.toml (см. «Печать в Electron-приложениях»):
#   [output] type_delay_ms = 10
#   [output] pre_recording_command = "$HOME/.local/bin/voxtype-target remember"
# собрать whisper-server (см. «Сборка whisper-server»)
voxtype-stt gigaam
```

Требуются `voxtype` с включённым remote-режимом, `curl`, `ffmpeg` не нужен. Веса моделей
скачиваются при первом запросе в кэш Hugging Face (дальше сервер работает офлайн,
`HF_HUB_OFFLINE=1` в юните).

Модели: [istupakov/gigaam-v3-onnx](https://huggingface.co/istupakov/gigaam-v3-onnx)
(`gigaam-v3-e2e-rnnt`) — ONNX-экспорт [GigaAM v3](https://github.com/salute-developers/GigaAM);
[istupakov/parakeet-tdt-0.6b-v3-onnx](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx)
— ONNX-экспорт NVIDIA Parakeet TDT 0.6B v3. Запускаются через
[onnx-asr](https://github.com/istupakov/onnx-asr).

### Сборка whisper-server

Второе мнение использует `whisper-server` из whisper.cpp с Vulkan. Модель — тот же
`ggml-large-v3-turbo.bin`, что и у локального профиля voxtype. Исходники собираются один раз:

```sh
# Заголовки Vulkan/SPIR-V без pacman (нет прав/sudo не нужен):
cd ~/.local/share
git clone --depth 1 https://github.com/KhronosGroup/Vulkan-Headers.git
git clone --depth 1 https://github.com/KhronosGroup/SPIRV-Headers.git
cd SPIRV-Headers && cmake -B build -DCMAKE_INSTALL_PREFIX=$HOME/.local && cmake --install build

cd ~/.local/share && git clone --depth 1 https://github.com/ggml-org/whisper.cpp.git
cd whisper.cpp && git submodule update --init --recursive --depth 1
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON -DWHISPER_BUILD_TESTS=OFF \
  -DVulkan_INCLUDE_DIR=$HOME/.local/share/Vulkan-Headers/include \
  -DVulkan_LIBRARY=/usr/lib/libvulkan.so.1 -DVulkan_GLSLC_EXECUTABLE=/usr/bin/glslc
cmake --build build --target whisper-server whisper-cli -j"$(nproc)"
```

Если cmake не находит `SPIRV-Headers`, добавить `-DCMAKE_PREFIX_PATH=$HOME/.local`;
если `ggml-vulkan` не видит `spirv/unified1/spirv.hpp` — `-DCMAKE_CXX_FLAGS=-I$HOME/.local/include`.
Проверка без voxtype:

```sh
~/.local/share/whisper.cpp/build/bin/whisper-cli -m ~/.local/share/voxtype/models/ggml-large-v3-turbo.bin -l ru файл.wav
```

## Переключение

```sh
voxtype-stt gigaam    # локальный сервер (remote mode) вместо whisper.cpp
voxtype-stt whisper   # вернуть локальный whisper.cpp, как было
voxtype-stt status    # что активно и живо ли
voxtype-stt toggle    # переключить туда-обратно
```

Переключатель меняет только ключи `[whisper]` в `~/.config/voxtype/config.toml` (значения
соседних профилей сохраняются в `~/.local/state/voxtype-stt/whisper-previous.env` и
восстанавливаются при возврате), затем перезапускает `voxtype.service`. Комментарии и все
остальные настройки файла не трогаются.

Сокет `voxtype-gigaam.socket` включён в `sockets.target`, поэтому после перезагрузки он
слушает, но сервис **не** стартует сам: первый запрос от voxtype поднимает его за ~1.4 с.
Когда профиль `whisper`, сервис просто остановлен и память не занимает.

## Печать в Electron-приложениях

Отдельная проблема — не распознавание, а **ввод текста**. В приложениях на Electron/Chromium
(Obsidian, VS Code, Discord) `wtype` теряет часть кириллических символов: он строит одну XKB-
раскладку на всю строку, и при большом наборе символов часть keysyms затирает друг друга.
В обычном терминале (foot, kitty) это незаметно, а в Obsidian из
«`я запустил сбор данных для подготовки... предсказаний`» получается
«`я запустил сборанныхля пготовки... прсказани`» — пропадают `д`, `л`, `м`, `ы`, `й`.

Замер на одном и том же предложении (вывод в Obsidian):

| способ | результат |
|---|---|
| одна длинная строка, `wtype -d 1` | теряет `д`, `л`, `м`, `ы`, `й` |
| одна длинная строка, `wtype -d 15` | теряет те же символы (задержка не помогает) |
| куски по ~40 символов, `-d 15` | всё равно теряет |
| **по одному слову, `-d 10`** | **точно** |

Оба условия важны: дробить строку (иначе ломается раскладка wtype) и держать задержку
между символами (иначе Electron не успевает).

Поэтому `wtype` обёрнут шимом `~/.local/share/voxtype-shim/wtype`, который:

1. держит focus guard — если фокус уехал из окна, где началась диктовка, текст летит в
   буфер обмена с уведомлением, а не в чужое окно;
2. когда окно то же, печатает текст **по одному слову** отдельными вызовами `/usr/bin/wtype`,
   сохраняя ключи, которые передал voxtype (`-d N`, `-s N`).

Шим подхватывается тремя частями:

* `~/.config/systemd/user/voxtype.service.d/path.conf` ставит каталог шима первым в `PATH`;
* `~/.local/bin/voxtype-target remember` вызывается через
  `pre_recording_command` в `~/.config/voxtype/config.toml` и запоминает окно диктовки;
* `type_delay_ms = 10` в `[output]` — задержка между символами (в voxtype 1.1.0 этот ключ
  убрали из `voxtype config`, но поле все ещё читается из файла и передаётся в `wtype` как `-d`).

Многострочный текст (переносы строк) шим отправляет одним вызовом, как раньше, — чтобы не
потерять семантику `shift_enter_newlines` и автосабмита.

## Смешанная речь и технические термины

Английские слова внутри русской фразы — самая слабая точка GigaAM. На фразе «Сейчас у меня 5
workspace'ов и ещё нулевой, на котором dashboard» три модели дали:

| модель | текст |
|---|---|
| GigaAM | Сейчас у меня 5 **wark space** и ещё нулевой, на котором **дашбор**. |
| whisper large-v3-turbo (локально) | сейчас у меня 5 **work space of** и еще **0** на котором **дашь board** и |
| Parakeet v3 | Сейчас у меня **пять воркспейсов** и еще нулевой, на котором **дашборд**. |

Parakeet переводит термины в кириллицу («воркспейсов», «дашборд»). Whisper на этой фразе тоже
ошибся, но на других дампах живой речи он единственный из трёх, кто вытягивает английские слова
внутри русской фразы:

| GigaAM слышит | whisper слышит |
|---|---|
| `Dipsic` | `DeepSeek` |
| `IPI от Dipsic` | `API Key от DeepSeek` |
| `Rast и Tipecript ... внутри Макс` | `Rust и TypeScript ... внутри Emacs` |
| `Doker Soket` | `Docker socket` |
| `VoxTipe` | `VoxType` |
| `CP` | `CPU` |

Обратная сторона: whisper иногда переводит английское слово в кириллицу, как Parakeet
(`юзер ID` → `юзер-ид`, `Dipsic Flash` → `Дипсик флэш`). Поэтому его результат берётся только
когда латинских слов у него не меньше, чем у GigaAM: иначе остаётся вывод GigaAM, который
можно поправить словарём замен. Задержка второго мнения — ~1.8 с и только на смешанных и
английских фразах.

Чтобы термины возвращались в латиницу, есть штатный словарь замен voxtype
(`text.replacements`): сопоставление целыми словами, без учёта регистра, литеральное.

Итог по фразе выше сейчас: «Сейчас у меня пять workspace'ов, и еще нулевой, на котором
dashboard.»

### Словарь замен

Набор правил лежит в `~/.local/share/voxtype-gigaam/replacements.tsv` (139 активных правил),
а не в `config.toml` — так его видно целиком, можно резать по секциям и откатывать одной
командой. Секции:

| секция | что внутри | правил |
|---|---|---|
| `numbers` | «пять» → `5`, до миллиона | 36 |
| `percent` | «процентов» → `%` | 6 |
| `tech-names` | `гитхаб` → `GitHub`, `кафку` → `Kafka`, `омарчи` → `Omarchy` и т. д. | 53 |
| `dev-slang` | `воркспейсов` → `workspace'ов`, `дашборд` → `dashboard`, `бэкенд` → `backend` | 38 |
| `punctuation` | «точка с запятой» → `;`, «новая строка» → перевод строки | 6 |
| `optional` | сомнительное (`один` → `1`, `точка` → `.`, `логи` → `logs`) — закомментировано | 13 |

```sh
voxtype-replacements               # применить всё из файла в config.toml
voxtype-replacements list           # что есть в файле
voxtype-replacements show           # что сейчас в config.toml
voxtype-replacements diff           # сравнить файл и config
voxtype-replacements remove         # убрать всё из файла
voxtype-replacements apply --section tech-names          # только одна секция
voxtype-replacements apply --except dev-slang             # всё, кроме секции
```

Правка словаря: добавить строку `найдено<TAB>замена` в нужную секцию, потом `voxtype-replacements`
и `systemctl --user restart voxtype.service`. В значениях можно писать `\n` — станет переводом
строки. Два ограничения voxtype, о которых стоит помнить: падежные формы перечисляются
отдельно (совпадение только по целому слову) и порядок правил не определён — нельзя заводить
одновременно «точка» и «точка с запятой», одна сломает другую.

Готовых публичных списков для русского не нашлось. В доках voxtype и в конфигах сообщества
(посмотрел десяток репозиториев) есть только примеры вида `"vox type" = "voxtype"`,
`javascript = "JavaScript"` — английские или точечные. Русский набор собран здесь с нуля.

## Диагностика

```sh
curl -s http://127.0.0.1:9017/health
curl -s http://127.0.0.1:9018/health              # whisper-server
journalctl --user -u voxtype-gigaam.service -f
journalctl --user -u whisper-server.service -f
journalctl --user -u voxtype.service -f          # видно "Sending ... to remote server"
~/.local/share/voxtype-gigaam/.venv/bin/python ~/.local/share/voxtype-gigaam/server.py --selftest <файл.wav>
```

Тест без микрофона — прогнать готовый WAV через активный профиль voxtype:

```sh
voxtype --quiet transcribe ~/.local/share/voxtype-gigaam/samples/ru_sample.wav
```

Если сервис не поднимается, в первую очередь проверять `journalctl --user -u voxtype-gigaam.service`:
он падает с внятной ошибкой, если весов нет в кэше Hugging Face, а `HF_HUB_OFFLINE=1` в юните
специально запрещает тихую сетевую догрузку.

Важно: `whisper.*` и `text.*` читаются демоном voxtype при старте, поэтому после правки
`config.toml` вручную нужен `systemctl --user restart voxtype.service`. Перезапуск только
`voxtype-gigaam.service` ничего не поменяет — демон продолжит слать старое значение
`whisper.remote_model` (именно так легко потерять автоматический выбор модели).
Скрипт `voxtype-stt` перезапускает voxtype сам.

### Диагностика точности

Сервер сохраняет аудио каждого запроса в `~/.local/state/voxtype-stt/dumps/` (WAV + JSON с
текстом, `rms`, `peak` и временем декода), последние 200 запросов. Это нужно, чтобы разбирать
ошибки распознавания на реальной речи, а не на синтетике. Вместе с каждой строкой лога
печатается `rms` и `peak`; если сигнал упирается в ±1.0, в лог попадает предупреждение о
клиппинге — тогда стоит снизить усиление микрофона (`pactl set-source-volume @DEFAULT_SOURCE@ 70%`).

Сравнить записанное аудио на всех моделях:

```sh
analyze-dumps                                   # все дампы, свежие первыми
analyze-dumps ~/.local/state/voxtype-stt/dumps '20260928-1522*.wav'
```

Что уже показывали эти замеры на живой речи:

* **GigaAM int8 против fp32** — одинаковый текст, одинаковая скорость (0.2 с на фразу 2.5 с).
  fp32-энкодер (885 МБ) не даёт выигрыша, поэтому используется int8.
* **Parakeet v3 на русском хуже GigaAM**: теряет точки (`Оно есть большие языковые модели`)
  и склеивает слова (`Большиековые`). Поэтому русская речь всегда идёт через GigaAM, а
  Parakeet остаётся только запасным вариантом, если whisper-сервер недоступен.
* **Сравнение по уверенности не работает.** На 126 фразах, которые получали второе мнение,
  Parakeet был «увереннее» GigaAM в 107 случаях — в среднем на +0.09, потому что шкалы
  логвероятностей у моделей разные. В 90 из этих 107 случаев вывод был русским, то есть
  русская речь уходила мультиязычной модели. Теперь решение принимается по письменности
  вывода GigaAM, а не по числу.
* **Whisper large-v3-turbo на смешанной речи.** На дампах живой диктовки он один
  вытягивает английские слова в латинице: `Rust и TypeScript ... внутри Emacs`,
  `API Key от DeepSeek`, `Docker socket`, `VoxType`. Он же может и транслитерировать
  (`юзер ID` → `юзер-ид`), поэтому берётся только если латинских слов у него не меньше,
  чем у GigaAM. Декод — ~1.8 с на фразу при загруженной модели.

## Тонкая настройка

- Порт: `VOXTYPE_GIGAAM_PORT` и `ListenStream` в socket-юните, затем `remote_endpoint` в
  voxtype (без `/v1` — voxtype сам дописывает `/v1/audio/transcriptions`).
- Язык: сервер сам решает по выводу русской модели. Один декод на русскую фразу, два —
  на смешанную или английскую. Принудительно выбрать модель можно только запросом с другим
  полем `model` (`gigaam-v3-e2e-rnnt` или `nemo-parakeet-tdt-0.6b-v3`); voxtype шлёт то,
  что стоит в `whisper.remote_model` (`auto`).
- Второе мнение через whisper: `VOXTYPE_WHISPER_ENDPOINT` в юните gigaam (по умолчанию
  `http://127.0.0.1:9018/inference`). Пусто — whisper не используется, английские и смешанные
  фразы падают на Parakeet. Таймаут — `VOXTYPE_WHISPER_TIMEOUT` (30 с).
- Parakeet как запасной движок: `VOXTYPE_GIGAAM_ENGLISH=0` в юните полностью выключает его
  (тогда при недоступном whisper остаётся только GigaAM).
- Словарь замен: `~/.local/share/voxtype-gigaam/replacements.tsv` + `voxtype-replacements`.
- Потоки ONNX Runtime: `VOXTYPE_GIGAAM_THREADS` в юните (по умолчанию 6 из 12).
- Чанки eager **выключены** (`whisper.eager_processing = false`), и это осознанно: склейка
  чанков в voxtype (`src/eager.rs`, `deduplicate_boundary`) сравнивает слова через
  `eq_ignore_ascii_case`, который фолдит только латиницу. Русские слова с заглавными и
  пунктуацией на границе 5-секундного чанка не совпадают, и слово печатается дважды.
  Включать обратно (`voxtype config set whisper.eager_processing true`) имеет смысл только
  после правки этой дедупликации; выигрыш при этом небольшой — GigaAM декодирует быстрее
  реального времени, так что задержка после отпускания клавиши и без eager меньше секунды.
- VAD **включён**: `vad.enabled = true`, `vad.backend = "energy"`, `vad.threshold = 0.3`
  (энергопорог ~0.004 RMS, то есть −48 dBFS). Записи без речи и тихий шум отсекаются до
  распознавания, поэтому GigaAM не выдумывает фразы из тишины. Проверено на файлах: тишина и
  шум с амплитудой 0.002 отклоняются (`No speech detected`), речь распознаётся. Если начнёт
  терять очень тихую речь — `voxtype config set vad.threshold 0.2`; выключить совсем —
  `voxtype config set vad.enabled false`, затем перезапуск `voxtype.service`. Бэкенд `energy`
  выбран сознательно: `auto` для движка `whisper` берёт Silero, а для неё нужна отдельная
  модель (`voxtype setup vad`).
- Прижимание моделей к памяти: `systemctl --user stop voxtype-gigaam.service whisper-server.service`
  освобождает ~520 МБ у GigaAM и ~1.6 ГБ у whisper-server; следующий запрос поднимет GigaAM
  снова (сокет-активация), а whisper-server — при следующем смешанном/английском запросе,
  если включён `WantedBy=graphical-session.target` (иначе — стартом юнита).

## Ограничения

- **Нет живого текста во время речи.** Есть только задержка после отпускания клавиши
  (~0.05x от длительности фразы). Настоящие partials (как в streaming zipformer) voxtype умеет
  только для `whisper` в локальном режиме, `parakeet.streaming` и `openvino.streaming`; для
  remote-режима нужен патч в `create_transcriber`.
- **`eager_processing` даёт дубли слов на стыках чанков** (см. «Тонкая настройка»).
- **whisper иногда транслитерирует английские слова.** `юзер ID` → `юзер-ид`,
  `Dipsic Flash` → `Дипсик флэш`. Лечится правилом: вывод whisper берётся только если
  латинских слов в нём не меньше, чем у GigaAM; иначе остаётся GigaAM, который можно
  поправить словарём замен.
- **Whisper пунктуацией слабее GigaAM.** В смешанной фразе можно получить «модели, по-моему»
  вместо «модели. По-моему». Зато английские слова не превращаются в «languigh».
- **Язык определяется по выводу, а не по аудио.** Чисто английская речь уходит на whisper, а
  если он недоступен — на Parakeet.
- **whisper-server держит ~1.6 ГБ RAM (модель в VRAM) постоянно.** Остановить вместе со всем:
  `systemctl --user stop voxtype-gigaam.service whisper-server.service`.
- **Нет перевода.** `whisper.translate` в remote-режиме отправляет запрос на
  `/v1/audio/translations`, а сервер его не реализует (вернёт текст без перевода).

## Удаление

```sh
voxtype-stt whisper
systemctl --user disable --now voxtype-gigaam.socket whisper-server.service
rm -f ~/.config/systemd/user/voxtype-gigaam.{socket,service}
rm -f ~/.config/systemd/user/whisper-server.service
rm -rf ~/.local/share/voxtype-gigaam ~/.local/state/voxtype-stt
rm -f ~/.local/bin/voxtype-stt
systemctl --user daemon-reload
```

Сборку whisper.cpp удалять не обязательно — она пригодится и другим способом
(`rm -rf ~/.local/share/whisper.cpp ~/.local/share/Vulkan-Headers ~/.local/share/SPIRV-Headers`,
если всё же нужно).

Кэш моделей Hugging Face удаляется отдельно:
`rm -rf ~/.cache/huggingface/hub/models--istupakov--{gigaam-v3-onnx,parakeet-tdt-0.6b-v3-onnx}`.
