<h1 align="center">Robotic Backgammon</h1>

<p align="center">
  <b>A real backgammon board that plays against you, or lets someone play from a phone anywhere in the world.<br>
  A robot hidden under the glass moves the checkers.</b>
</p>

<p align="center">
  <img src="docs/media/hero-remote-move.gif" width="860" alt="A move tapped on a phone is played by the robot on the real board">
</p>

<p align="center">
  ESP32-S3 &nbsp;·&nbsp; CoreXY &nbsp;·&nbsp; 12 V electromagnet &nbsp;·&nbsp; camera through a mirror &nbsp;·&nbsp; Python &nbsp;·&nbsp; GNU Backgammon
</p>

---

## How it moves

A cart on two rails runs under 4 mm tempered glass. It carries an electromagnet that grabs a checker from below and drags it to its place, along a path that keeps clear of every other checker.

<table>
  <tr>
    <td width="33%" align="center"><img src="docs/media/magnet-under-glass.gif" alt="The electromagnet drags a checker under the glass"></td>
    <td width="33%" align="center"><img src="docs/media/photo-magnet-carriage.jpg" alt="The magnet cart on the CoreXY rails"></td>
    <td width="33%" align="center"><img src="docs/media/photo-checkers.jpg" alt="3D-printed checkers"></td>
  </tr>
  <tr>
    <td align="center"><sub><b>Under the glass.</b> The magnet pulls the checker along.</sub></td>
    <td align="center"><sub><b>The cart.</b> CoreXY, two NEMA 17 motors, MGN12 rails, GT2 belts.</sub></td>
    <td align="center"><sub><b>The checkers.</b> 3D-printed, 26 mm, with a steel bottom.</sub></td>
  </tr>
</table>

## How it sees

Most robot boards hang a camera above the table. This one keeps it inside the box: the camera sits at the centre of the back wall and looks into a mirror, up through the glass. The software straightens the view into a top-down image of the board.

<p align="center">
  <img src="docs/media/cad-camera-mirror.jpg" width="760" alt="Section of the box: camera, mirror and glass">
</p>
<p align="center">
  <img src="docs/media/camera-view.jpg" width="860" alt="The raw camera image and the straightened top view">
</p>

## It sets itself up

From two rows of checkers in the trays, the robot lays out the opening position on its own (sped up).

<p align="center">
  <img src="docs/media/auto-setup.gif" width="440" alt="The robot sets up the opening position by itself">
</p>

## The electronics

<p align="center">
  <img src="docs/media/wiring-overview.png" width="900" alt="Wiring overview">
</p>

An ESP32-S3 drives two TMC2209 stepper drivers on a CNC Shield and switches the magnet through an opto-isolated MOSFET module. The PC talks to it over USB serial. The lid, with the controller and camera, unplugs from the box through an RJ45 and a banana connector. Every pin and wire is in the [detailed wiring diagram](electronics/wiring_diagram.png) and [wiring notes](electronics/wiring.md).

## Designed in Fusion 360

<table>
  <tr>
    <td width="45%" align="center"><img src="docs/media/cad-assembly.gif" alt="The Fusion 360 model assembling"></td>
    <td>
      Every part was modelled before it was built: the box, the CoreXY frame, the cart, the mirror and the camera mount.
      <br><br>
      <a href="https://a360.co/4yloJTi"><b>Open the 3D model in your browser</b></a>
    </td>
  </tr>
</table>

## Ways to play

* **Against the computer.** GNU Backgammon picks the robot's moves, and the robot plays them on the board.
* **From far away.** A second player gets a private link and plays from a phone; the robot moves their checkers at home.
* **Two people at the board,** with the robot keeping track.
* **Robot against robot,** as a demo.

You play your own checkers by hand. The game runs in the browser on the PC (`pc/game.py`), with digital dice, an undo button and an advantage bar, and it also runs as a simulator with no hardware attached.

## Under the hood

| Part | What it is |
|---|---|
| Mechanics | CoreXY, MGN12 linear rails, GT2 belts, 2 × NEMA 17, wooden box, 4 mm tempered glass |
| Actuator | P30/25 12 V electromagnet, AOD4184 MOSFET module, flyback diode, heat budget enforced by the firmware |
| Controller | Freenove ESP32-S3 WROOM CAM, CNC Shield V3, 2 × TMC2209, 12 V 8 A supply |
| Link | USB serial at 921600 baud; one command per line in millimetres, answered `DONE` or `ERR` |
| Vision | OV3660 wide-angle camera looking into a mirror; calibrated by the robot itself, which carries a checker over a grid and photographs it |
| Software | Python on the PC: game logic, browser UI, GNU Backgammon engine, A* drag paths that keep 30 mm from other checkers, remote play over a secure tunnel |

## Repository

| Folder | Contents |
|---|---|
| [`pc/`](pc) | Everything that runs on the PC: the game, calibration and setup tools ([details](pc/README.md)) |
| [`firmware/`](firmware) | ESP32-S3 firmware: motion, homing, magnet, camera ([details](firmware/README.md)) |
| [`electronics/`](electronics) | Wiring diagram, wiring notes and wiring table |

<details>
<summary><b>The original build plan</b></summary>

<p align="center">
  <img src="robot_backgammon_preview.png" width="600" alt="Early system overview">
  <br><em>An early render: the CoreXY mechanism beneath the board</em>
</p>

```mermaid
graph TD
    classDef hardware fill:#ffdfba,stroke:#ffb347,stroke-width:2px,color:#000,rx:8,ry:8,font-weight:bold;
    classDef mech fill:#baffc9,stroke:#4caf50,stroke-width:2px,color:#000,rx:8,ry:8,font-weight:bold;
    classDef software fill:#bae1ff,stroke:#2196f3,stroke-width:2px,color:#000,rx:8,ry:8,font-weight:bold;
    classDef integration fill:#ffffba,stroke:#ffeb3b,stroke-width:2px,color:#000,rx:8,ry:8,font-weight:bold;
    
    subgraph Legend ["<b style='font-size:18px;'>📌 Legend - Color Code</b>"]
        direction LR
        L1["🔌 Hardware & Wiring"]:::hardware
        L2["⚙️ Mechanics & CAD"]:::mech
        L3["💻 Software & AI"]:::software
        L4["📦 Integration & Build"]:::integration
    end

    subgraph P1 ["<b style='font-size:20px;'>🚀 Phase 1: Procurement</b>"]
        A1["🛒 1. Initial Procurement"]:::hardware
    end

    subgraph P2 ["<b style='font-size:20px;'>⚙️ Phase 2: Core Mechanics & Prototyping</b>"]
        B1["🖨️ 2. Print CoreXY Parts"]:::mech
        B2["🪵 3. Mount on Temp Wood"]:::mech
        B3["⚡ 4. Wire & Test 1 Motor"]:::hardware
        B4["📏 5. Measure Actual Z-Gap"]:::mech
    end

    subgraph P3 ["<b style='font-size:20px;'>♟️ Phase 3: Game Pieces & Magnet</b>"]
        C1["🧲 6. Print 1 Piece & Test Drag"]:::mech
        C2["🎲 7. Print All Game Pieces"]:::mech
        C3["📐 8. Design Magnet Mount"]:::mech
    end

    subgraph P4 ["<b style='font-size:20px;'>🔌 Phase 4: Electronics & Firmware</b>"]
        D1["🔋 9. Full Wiring & MOSFET"]:::hardware
        D2["💻 10. ESP32 Motion Code"]:::software
        D3["📡 11. Setup ESP32 OTA"]:::software
    end

    subgraph P5 ["<b style='font-size:20px;'>🧠 Phase 5: Vision & Logic</b>"]
        E1["💬 12. Basic Comm: Direct Command"]:::software
        E2["👁️ 13. Train YOLO on Pieces"]:::software
        E3["📱 14. Build UI/App Display"]:::software
        E4["🤖 15. Backgammon Game Logic"]:::software
    end

    subgraph P6 ["<b style='font-size:20px;'>📦 Phase 6: Final Enclosure</b>"]
        F1["📐 16. Decide Final Dimensions"]:::integration
        F2["🪟 17. Order Perspex Board"]:::hardware
        F3["🪚 18. Build Wooden Box"]:::integration
        F4["🎉 19. Full System Integration"]:::integration
    end

    A1 --> B1
    A1 --> C1
    B1 --> B2
    B2 --> B3
    B3 --> B4
    C1 --> C2
    B4 --> C3
    B4 --> F1
    F1 --> F2
    F1 --> F3
    C3 --> D1
    B3 --> D1
    D1 --> D2
    D2 --> D3
    D2 --> E1
    C2 --> E2
    E2 --> E3
    E1 --> E4
    E2 --> E4
    D3 --> F4
    E4 --> F4
    F2 --> F4
    F3 --> F4
    
    style Legend fill:#ffffff,stroke:#333,stroke-width:2px,stroke-dasharray: 5 5,rx:10,ry:10
    style P1 fill:#f9f9f9,stroke:#ccc,stroke-width:3px,rx:10,ry:10
    style P2 fill:#f9f9f9,stroke:#ccc,stroke-width:3px,rx:10,ry:10
    style P3 fill:#f9f9f9,stroke:#ccc,stroke-width:3px,rx:10,ry:10
    style P4 fill:#f9f9f9,stroke:#ccc,stroke-width:3px,rx:10,ry:10
    style P5 fill:#f9f9f9,stroke:#ccc,stroke-width:3px,rx:10,ry:10
    style P6 fill:#f9f9f9,stroke:#ccc,stroke-width:3px,rx:10,ry:10
```

</details>
