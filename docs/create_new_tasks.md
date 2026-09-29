# Neue sequentielle Tasks auf Basis von `common/sequential_task`

## 1. Überblick über die Files

| Datei | Inhalt |
|---|---|
| `sequential_task_config.py` | hält alle notwendigen Configs aus denen einmalig zu Beginn alle Spezifikationen von Entitäten im Modell ausgelesen werden und Konfigurationen für die Env als auch die TaskLogic bereitgestellt werden : `SequentialTaskConfig`, `PointingTargetConfig`, `ButtonTargetConfig`, `ReachConfig`, `ButtonPressConfig`, `RewardConfig`, `Color_mode`, … |
| `sequential_task_component.py` | `SequentialTaskComponent`: baut das MuJoCo-Modell inkl. Targets und verdrahtet Observations, Actions, Rewards, Events, Terminations, Metrics und Visualisierung |
| `sequential_task_logic.py` | `SequentialTaskLogic`: Step-Event, das Sequenz, Dwell-Zeit und Phasenfortschritt verwaltet |
| `sequential_task_entity.py` | `TaskEntity` / `TaskEntityCfg`: Scene-Entity, auf der der gesamte Task-Zustand als Tensoren liegt |
| `sequential_task_mdp.py` | Observation-, Reward- und Terminierungsfunktionen |
| `target_coloring.py` | Einfärbung der Targets nach Fortschritt |
| `target_resolver.py` | OmegaConf-Resolver `${task_target:pointing}` / `${task_target:button}` |

Grundidee: Eine Episode besteht aus `num_phases` **Phasen**. In jeder Phase ist
genau ein Target aus einem **Target-Pool** (`cfg.targets`) aktiv. Ist das
aktive Target „erfüllt“ (lange genug „inside“), springt die Sequenz zur
nächsten Phase. Nach der letzten Phase endet die Episode erfolgreich.

## 2. Was damit schon implementiert ist

### 2.1 Modellaufbau (MuJoCo-Spec)

`SequentialTaskComponent._create_model()` erledigt:

* Laden des Modells aus `cfg.model_path` und einmaliges Kompilieren +
  `mj_forward`, um Namen (Joints, Tendons, Actuators) und die Weltposition der
  Referenz-Site zu bestimmen.
* **Endeffektor zum Interagieren**: Weltposition von `reach.reference_site` (Standard
  `humphant`) in der Default-Pose + `reach.reference_offset`.
* **Pointing-Targets** (`PointingTargetConfig`): je ein Body
  `body_target_{i}` mit einer Geom `geom_target_{i}` (Kugel oder Box) mit
  MuJoCo-Default-Kontaktparametern (`contype`/`conaffinity` werden nicht
  gesetzt); Kontakte werden von der Logik nicht ausgewertet.
* **Button-Targets** (`ButtonTargetConfig`), zwei Varianten:
    * *Touch-Button* (`button_press.enabled = false`): kollidierende Box
      `geom_target_{i}` mit Rotation `euler`, Touch-Site `site_target_{i}`
      und Touch-Sensor `sensor_target_{i}`.
    * *Drückbarer Button* (`button_press.enabled = true`): unsichtbare
      Referenz-Box `geom_target_{i}` (hält Masse/Mittelpunkt und Größe für die
      Observations), sichtbares Gehäuse aus 5 Geoms
      `geom_housing_{i}_{k}`, beweglicher Knopf `body_button_cap_{i}` an einem
      gefederten Slide-Joint `joint_button_{i}` (Federsteifigkeit, Vorspannung,
      Dämpfung, harte Anschläge) plus Touch-Site und -Sensor auf dem Knopf.
      Geometrische Plausibilitätschecks (Knopf passt ins Gehäuse, reicht nicht
      durch die Rückplatte) inklusive.
* Deaktivieren von `multiccd`, sobald Buttons vorkommen (sonst lehnt
  `mujoco_warp` die Box-Mesh-Paare ab).

### 2.2 Positions Randomization der Targets

Jede Target-Pos ist ein `Vec3Range`/`ScalarRange`: entweder ein fester
`value` **oder** ein Intervall `min`/`max`. Bei Intervallen wird automatisch ein
Reset-Event angelegt:

| Eigenschaft | Event | Anmerkung |
|---|---|---|
| `position` | `target_pos_dr` (`dr.body_pos`) | pro Achse gleichverteilt, relativ zum Target-Ursprung |
| `size` | `target_size_dr` (`dr.geom_size`) | Pointing: jede der 3 Achsen unabhängig aus `[min, max]` |
| `rgb` | `target_rgba_dr` (`dr.geom_rgba`) | **Default ist zufällig** (`min=(0,0,0)`, `max=(1,1,1)`) |

Die Randomisierung passiert pro Environment bei jedem Reset.

### 2.3 Scene-Entity und Task-Zustand

`TaskEntityCfg` registriert Arm + Targets als **eine** Entity (Name =
`entity_name` der Komponente) inkl. Tendon-Actuators. Auf der `TaskEntity`
liegt der komplette Zustand als batched Tensoren (`[num_envs, …]`), u. a.
`phase_targets`, `current_phase`, `current_target_id`, `target_pos`,
`distance_to_target`, `inside_target`, `steps_inside_target`,
`current_phase_completed`, `completed_target_count`,
`remaining_target_distances`. Eigene Observations/Rewards können direkt darauf
zugreifen.

### 2.4 Sequenz- und Phasenlogik (`SequentialTaskLogic`)

Läuft als Step-Event (`task_logic_update`) und als Reset-Hook:

* **Reset**: Phase, Zähler und Flags zurücksetzen, neue Sequenz
  `phase_targets` ziehen (`_sample_phase_targets`), Target-Positionen aus der
  Simulation lesen und die Länge des noch zurückzulegenden Weges zum Target
  entlang der Sequenz (`remaining_target_distances`) vorberechnen.
* **Step**:
    1. Wurde im letzten Step eine Phase abgeschlossen, Zähler erhöhen und zur
       nächsten Phase wechseln (Dwell-Zähler zurücksetzen).
    2. `_inside_target()` auswerten → `inside_target`, `distance_to_target`,
       `target_size`.
    3. Optional Release-Pflicht für Buttons (`button_press.require_release`):
       Nach Abschluss muss ein noch gedrückter Button erst losgelassen werden,
       bevor er erneut zählt (wichtig, wenn derselbe Button zweimal in Folge
       kommt).
    4. Dwell-Zähler erhöhen; bei `reach.dwell_continuous = true` wird er beim
       Verlassen des Targets auf 0 gesetzt, sonst **akkumuliert** er.
    5. `current_phase_completed = steps_inside_target >= dwell_steps`.
* **Erfüllungskriterien** (Default-`_inside_target`):
    * Pointing: euklidische Distanz Endeffektor-Site ↔ Target-Mittelpunkt
      `< geom_size[0]` (Radius).
    * Touch-Button: Touch-Sensor `>= min_touch_force`.
    * Drückbarer Button: Eindrücktiefe des Knopfs `>= activation_depth`.
* **Dwell-Zeit**: `max(1, ceil(dwell_duration / step_dt))` Steps pro Target,
  d. h. jedes Target muss mindestens einen Step lang erfüllt sein.

### 2.5 Observations

Zwei Gruppen, `agent_state` und `task_state`. Welche Terme enthalten sind,
steuern die Listen `agent_state_keys` / `task_state_keys` in der Config. Fertig
verfügbare Keys:

| Key | Bedeutung |
|---|---|
| `qpos` | unabhängige Gelenkwinkel, auf `[-1, 1]` normiert |
| `qvel`, `qacc` | Gelenkgeschwindigkeit / -beschleunigung |
| `act` | Muskelaktivierungen, auf `[-1, 1]` skaliert |
| `ee_pos` | Weltposition aller Sites aus `_site_names()` (konkateniert) |
| `target_pos` | Position des **aktuellen** Targets |
| `target_size` | `geom_size` (3 Werte) des aktuellen Targets |
| `phase_progress` | Phase auf `[-1, 1]` normiert |
| `dwell_fraction` | Anteil der erfüllten Dwell-Zeit |
| `button_press_fraction` | Eindrücktiefe / `travel` (0 für Nicht-Buttons) |
| `time` | siehe Stolperfallen (Abschnitt 5) |

Default: `agent_state = [qpos, qvel, qacc, act, ee_pos]`,
`task_state = [target_pos, target_size, phase_progress, dwell_fraction]`.


### 2.6 Rewards

Alle Terme werden immer angelegt, anschließend gewichtet aus `reward.weights` (fehlender Key
⇒ Gewicht 0):

| Key | Funktion |
|---|---|
| `distance` | −(Distanz zum aktuellen Target + Restweg über alle folgenden Targets der Sequenz). Mit `distance_exponential = true`: `(exp(−d·metric) − 1)/metric`, innerhalb des Targets 0 |
| `phase_bonus` | 1 im Step, in dem eine Phase abgeschlossen wird |
| `done` | 1, wenn die letzte Phase abgeschlossen ist |
| `dc_effort` | −0.1477 · ‖ctrl‖² |
| `jac_effort` | Effort + Gelenkbeschleunigungs-Kosten |

### 2.7 Events, Terminations, Metrics

* **Events**: `reset_joints` (Gelenkpositionen/-geschwindigkeiten ±0.1 um die
  Default-Pose), `task_logic_update` (Step), ggf. die DR-Events aus 2.2.
* **Terminations**: `time_out` und `episode_success` (letzte Phase erfüllt).
  Es gibt **keine** Fehlschlag-Terminierung.
* **Metrics**: `phase_{i}_success` für jede Phase, `inside_target`,
  `total_initial_distance`, `target_size`, `completed_target_count`.

### 2.8 Visualisierung 

`target_state_color_mode`:

* `OFF`: Originalfarben.
* `RECOLOR`: `geom_rgba` im Modell wird pro Step gesetzt (grün = aktuell, rot =
  noch offen, blau = erledigt). Führt in viser zu Lag.
* `OVERLAY`: gleiche Farben als Debug-Geometrie über den Targets, ohne
  Modelländerung (empfohlen; nur Kugel/Box-Targets unterstützt).

### 2.9 Konfiguration aus YAML

`targets` wird per `hydra.utils.instantiate` aus einer YAML-Liste erzeugt, die
Typen werden über den Resolver gewählt:

```yaml
targets:
  - _target_: ${task_target:button}     # oder ${task_target:pointing}
    label: "5"
    position:
      value: [0.40, -0.12, -0.20]        # oder min: [...] / max: [...]
    rgb:
      value: [0.2, 0.2, 0.2]
```

### 3 Minimales Rezept für eine neue Task

1. Ordner `myo_core/task/<name>/` anlegen.
2. Config: `@dataclass class <Name>TaskConfig(SequentialTaskConfig)` mit
   `model_path` und eigenen Feldern.
3. Optional Logik: Unterklasse von `SequentialTaskLogic`, nur die benötigten
   Hooks überschreiben.
4. Komponente: `@myo_register_task("<name>")` auf eine Unterklasse von
   `SequentialTaskComponent`; `__init__(self, cfg: <Name>TaskConfig)` mit
   **genau einem** typannotierten Argument (daraus leitet die Registry die
   Config-Klasse ab); `entity_name` und ggf. `task_logic_cls` setzen.
5. `__init__.py` des Pakets: Config + Komponente importieren und
   `register_target_resolver()` aufrufen.
6. In `myo_core/task/__init__.py` das Paket importieren (sonst wird die Task
   nicht registriert).
7. `myo_config/task/<name>.yaml` mit den Targets anlegen; Auswahl per
   `task=<name>`.

## 4. Vorannahmen und Einschränkungen

### 4.1 Sequenz und Reihenfolge

* **Starre Default-Reihenfolge**: Ohne Override wird jedes Target genau einmal
  in der Reihenfolge der YAML-Liste abgearbeitet; `num_phases = num_targets`.
* **Feste Sequenzlänge**: `num_phases` ist pro Task konstant (Tensorform) –
  gleich für alle Environments und Episoden. Variable Längen pro Episode sind
  nicht vorgesehen. Die Reihenfolge selbst darf pro Env und Reset zufällig
  sein (siehe Numpad, mit/ohne Zurücklegen).
* **Zwei Stellen für `num_phases`**: Komponente (`_num_phases`) und Logik
  (`_resolve_num_phases`) müssen **manuell konsistent** gehalten werden – die
  eine bestimmt Rewards/Terminierung/Metrics, die andere die Tensoren.
* **Strikt linear**: Es gibt immer genau ein aktives Target; keine parallelen,
  optionalen oder verzweigenden Ziele, kein Zurückspringen.
* **Falsche Targets sind folgenlos**: Nur das aktuelle Target wird geprüft.
  Berühren/Drücken anderer Targets wird weder bestraft noch beendet es die
  Episode.
* **Phasenwechsel mit einem Step Verzögerung**: Abschluss wird in Step *t*
  erkannt, der Wechsel zum nächsten Target erfolgt am Anfang von Step *t+1*.

### 4.2 Target-Typen

* Jede Target-Config **muss** `dwell_duration` besitzen (die Logik liest es für
  alle Targets).
* Box-Pointing-Targets werden in der Logik wie **Kugeln mit Radius
  `size[0]`** behandelt (euklidische Distanz).
* Pointing-Targets haben **keine Orientierung**; nur Buttons 
* Targets sind **statisch in der Welt** (direkt am Worldbody, keine Joints –
  bis auf den Knopf drückbarer Buttons). Bewegte Targets, Targets am Körper
  oder Distraktoren gibt es nicht.
* Der Distanz-Reward zielt auch bei Buttons auf den Mittelpunkt des
  Button-Bodys, nicht auf die Oberfläche.
* `button_press` gilt **global** für alle Buttons einer Task (nicht pro
  Button).
### 4.3 Modell und Benennung
* Target-Positionen werden nur beim **Reset** gelesen; ändern sie sich während
  der Episode, sind `target_pos` und `remaining_target_distances` veraltet.

## 5. Stolperfallen

* **Unbekannte Keys werden ohne Fehlermeldung ignoriert**: Tippfehler in
  `agent_state_keys`, `task_state_keys` oder `reward.weights` führen zu
  fehlenden Termen bzw. Gewicht 0 ohne Fehlermeldung.
* **`ee_pos`** konkateniert *alle* Sites aus `_site_names()`. Wer weitere Sites
  hinzufügt, verändert damit die Observation-Dimension.
* **`time`-Observation**: Der Term wird ohne `asset_cfg` angelegt, die
  Funktion `myo.time` erwartet aber eines – vor Benutzung prüfen/fixen.
* **Nicht implementierte Config-Felder**: `distractor` (`DistractorConfig`)
  und `sequence` (`show_future_targets`, `color_code_progress`) existieren
  nur in der Config und haben noch keinen Effekt (legacy alte Version Universal). Muss selber implementiert werden, wenn notwendig
* `site_pos`/`site_size` von Buttons werden **nicht** randomisiert und auf die im Yaml definierte Pos gesetzt
* Bei drückbaren Buttons wirkt Farb- und Größen-Randomization auf die unsichtbare
  Referenz-Box
