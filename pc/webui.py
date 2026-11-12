"""The game page: a small web server (Python standard library only) around game.Game.

    py game.py      (starts this, opens the browser)

GET  /           the page
GET  /state      the game as JSON (the page asks twice a second)
GET  /photo.jpg  the last photo with what vision found
POST /action     {"name": "new" | "done" | "dice" | ..., "value": ...}

The terminal works too (for when the page is not open): type
  n = new game   d = done   a = accept   r = read again   c = closest legal move
  52 = dice 5-2  m 24/18 13/11 = pick a move   q = quit
"""
import json
import socket
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from game import DEFAULTS, Game

PORT = 8765

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI-bg</title>
<style>
:root{--bg:#f4efe6;--card:#fffaf2;--ink:#2b2622;--mute:#7a6f64;--line:#d9cbb5;
 --robot:#f2f2ea;--human:#b0302a;--accent:#2f6f5e;--warn:#b5651d}
@media (prefers-color-scheme:dark){:root{--bg:#1d1b19;--card:#27241f;--ink:#eee6da;--mute:#a3988a;
 --line:#4a4236;--robot:#e9e6dc;--human:#c8433b;--accent:#5bb39a;--warn:#e09a4f}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
 font:15px/1.4 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1180px;margin:0 auto;padding:16px;display:grid;gap:16px;
 grid-template-columns:minmax(0,1.1fr) minmax(0,1fr)}
@media (max-width:820px){main{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px}
h1{font-size:18px;margin:0 0 4px}.mute{color:var(--mute);font-size:13px}
#prompt{font-size:20px;font-weight:600;margin:6px 0 10px}
button,select,input{font:inherit;border-radius:8px;border:1px solid var(--line);padding:8px 12px;
 background:var(--card);color:var(--ink)}
button{cursor:pointer;background:var(--accent);color:#fff;border:0;font-weight:600}
button.sec{background:transparent;color:var(--ink);border:1px solid var(--line)}
#asks{display:flex;flex-wrap:wrap;gap:8px;align-items:center}
.dice{display:inline-flex;gap:6px;margin-left:8px}.die{width:34px;height:34px;border-radius:7px;
 background:#fff;color:#222;display:grid;place-items:center;font-weight:700;border:1px solid #bbb}
svg{width:100%;height:auto;display:block}
.warn{color:var(--warn);font-size:13px}
table{width:100%;border-collapse:collapse;font-size:13px}td{padding:3px 4px;border-bottom:1px solid var(--line)}
#photo{width:100%;border-radius:8px;display:none;margin-top:8px}
.grid{display:grid;grid-template-columns:auto 1fr;gap:6px 10px;align-items:center}
.stats{display:flex;flex-wrap:wrap;gap:4px 16px;font-size:13px;margin-top:6px}
</style></head><body><main>
<section class="card">
  <h1>AI-bg <span class="mute" id="sim"></span></h1>
  <div id="prompt">…</div>
  <div id="asks"></div>
  <div class="stats"><span id="turn"></span><span id="pips"></span><span id="offs"></span></div>
  <div id="coach" class="mute"></div>
  <div id="warns"></div>
  <svg id="board" viewBox="-34 -20 478 490"></svg>
  <div class="mute">Seen from above, near side (bear-off) at the bottom. Numbers are your points.
  White = robot, red = you.</div>
</section>
<section style="display:grid;gap:16px;align-content:start">
  <div class="card"><h1>Settings</h1><div class="mute">Used from the next New game.</div>
   <div class="grid" id="settings" style="margin-top:8px"></div></div>
  <div class="card"><h1>Moves</h1><table id="hist"></table></div>
  <div class="card"><h1>Camera</h1><div class="mute">Last photo, circles = checkers found.</div>
   <img id="photo" alt="last photo"></div>
</section></main>
<script>
const OPTS={mode:["robot_vs_human","robot_vs_robot","watch"],engine:["gnubg","simple","random"],
 dice:["digital","typed","camera"],human_move:["done_button","auto","robot_moves"],
 vision:["scan","slots","hough"],coach:[false,true],first:["roll","robot","human"],board:["auto","real","sim"]};
const NAMES={mode:"Mode",engine:"Engine",dice:"Dice",human_move:"Your move",vision:"Vision",
 coach:"Coach",first:"Who starts",board:"Board"};
const BTN={new:"New game",done:"Done",accept:"Accept",reread:"Read again",closest:"Take closest legal move",
 learn:"Learn colours",arrange:"Robot arranges the board",rolled:"Rolled"};
let S=null, built=false, lastPhoto=0;
function post(a){return fetch("/action",{method:"POST",body:JSON.stringify(a)})}
function el(t,attrs={},kids=[]){const e=document.createElement(t);for(const k in attrs){
 if(k=="text")e.textContent=attrs[k];else if(k.startsWith("on"))e[k]=attrs[k];else e.setAttribute(k,attrs[k])}
 for(const c of kids)e.append(c);return e}
function buildSettings(s){const g=document.getElementById("settings");g.innerHTML="";
 for(const k in OPTS){const sel=el("select",{onchange:()=>{let v=sel.value;if(v==="true")v=true;if(v==="false")v=false;
   post({name:"settings",values:{[k]:v}})}});
  for(const o of OPTS[k]){const op=el("option",{value:String(o),text:String(o).replaceAll("_"," ")});
   if(String(s[k])===String(o))op.selected=true;sel.append(op)}
  g.append(el("label",{text:NAMES[k]}),sel)}}
function drawBoard(s){const svg=document.getElementById("board");const H=430;
 let h=`<rect x="-32" y="-16" width="474" height="482" rx="10" fill="var(--card)" stroke="var(--line)"/>`;
 h+=`<rect x="150" y="-10" width="110" height="${H-60}" fill="var(--line)" opacity=".25"/>`;
 h+=`<text x="205" y="${H-236+4}" font-size="12" text-anchor="middle" fill="var(--mute)">BAR</text>`;
 for(const [x,y,n] of s.labels){const left=x<205, X=left?x-14:x+14;
  const pts=left?`${x-13},${H-y-13} ${x-13},${H-y+13} ${x+150},${H-y}`:`${x+13},${H-y-13} ${x+13},${H-y+13} ${x-150},${H-y}`;
  h+=`<polygon points="${pts}" fill="${n%2?'#c9a77c':'#8f6a43'}" opacity=".45"/>`;
  h+=`<text x="${left?-22:432}" y="${H-y+4}" font-size="11" text-anchor="middle" fill="var(--mute)">${n}</text>`}
 h+=`<text x="205" y="${H-31-4}" font-size="11" text-anchor="middle" fill="var(--mute)">bear-off</text>`;
 for(const [x,y,side] of s.pieces){h+=`<circle cx="${x}" cy="${H-y}" r="12.5" fill="var(--${side})" stroke="#0006" stroke-width="1.2"/>`}
 svg.innerHTML=h}
function render(s){S=s;if(!built){buildSettings(s.settings);built=true}
 document.getElementById("sim").textContent=s.sim?"(simulator)":"";
 const p=document.getElementById("prompt");p.textContent=s.prompt;
 if(s.dice){const d=el("span",{class:"dice"});for(const v of s.dice)d.append(el("span",{class:"die",text:v}));p.append(d)}
 const a=document.getElementById("asks");const keep=document.activeElement&&a.contains(document.activeElement);
 if(!keep||a.dataset.k!==JSON.stringify(s.asks)+s.legal.length){a.innerHTML="";a.dataset.k=JSON.stringify(s.asks)+s.legal.length;
  for(const k of s.asks){
   if(k==="dice_input"){const i=el("input",{placeholder:"dice, e.g. 5 2",size:8});
    a.append(i,el("button",{text:"OK",onclick:()=>post({name:"dice",value:i.value})}))}
   else if(k==="move_pick"){const sel=el("select");for(const m of s.legal)sel.append(el("option",{text:m}));
    a.append(sel,el("button",{text:"Play this",onclick:()=>post({name:"move",value:sel.value})}))}
   else a.append(el("button",{text:BTN[k]||k,onclick:()=>post({name:k})}))}
  a.append(el("button",{class:"sec",text:"New game",onclick:()=>post({name:"new"})}))}
 document.getElementById("turn").textContent=s.turn?("Turn: "+s.turn):"";
 document.getElementById("pips").textContent=`Pips robot ${s.pips[0]} · you ${s.pips[1]}`;
 document.getElementById("offs").textContent=`Off robot ${s.off[0]} · you ${s.off[1]}`;
 document.getElementById("coach").textContent=s.coach||"";
 const w=document.getElementById("warns");w.innerHTML="";for(const t of s.warnings)w.append(el("div",{class:"warn",text:"⚠ "+t}));
 drawBoard(s);
 const t=document.getElementById("hist");t.innerHTML="";
 for(const m of s.history.slice().reverse())t.append(el("tr",{},[el("td",{text:m.n}),el("td",{text:m.who}),
  el("td",{text:m.dice.join("-")}),el("td",{text:m.move}),el("td",{class:"mute",text:m.note||""})]));
 if(s.has_photo&&s.photo_t!==lastPhoto){lastPhoto=s.photo_t;const im=document.getElementById("photo");
  im.src="/photo.jpg?t="+s.photo_t;im.style.display="block"}}
async function tick(){try{render(await (await fetch("/state")).json())}catch(e){}setTimeout(tick,500)}
tick();
</script></body></html>"""


def make_handler(game):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body, ctype, code=200):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/":
                self._send(PAGE.encode(), "text/html; charset=utf-8")
            elif path == "/state":
                self._send(json.dumps(game.snapshot()).encode(), "application/json")
            elif path == "/photo.jpg" and game.photo:
                self._send(game.photo, "image/jpeg")
            else:
                self._send(b"not found", "text/plain", 404)

        def do_POST(self):
            if self.path != "/action":
                return self._send(b"not found", "text/plain", 404)
            n = int(self.headers.get("Content-Length", 0))
            try:
                game.post(json.loads(self.rfile.read(n) or b"{}"))
            except ValueError:
                return self._send(b"bad json", "text/plain", 400)
            self._send(b"{}", "application/json")
    return H


def terminal(game):
    import dice
    keys = {"n": "new", "d": "done", "a": "accept", "r": "reread", "c": "closest", "l": "learn",
            "g": "arrange", "o": "rolled"}
    while True:
        try:
            t = input().strip()
        except EOFError:
            return
        if t == "q":
            game.post({"name": "stop"})
            return
        if t in keys:
            game.post({"name": keys[t]})
        elif t.startswith("m "):
            game.post({"name": "move", "value": t[2:].strip()})
        elif dice.parse(t):
            game.post({"name": "dice", "value": t})
        elif t:
            print("n=new d=done a=accept r=read again c=closest l=learn g=arrange o=rolled 52=dice m <move> q=quit")


def lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "localhost"


def main(settings=None, open_browser=True, port=PORT):
    game = Game(settings)
    srv = ThreadingHTTPServer(("0.0.0.0", port), make_handler(game))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://localhost:{port}"
    print(f"Game page: {url}   (phone on the same WiFi: http://{lan_ip()}:{port})")
    print("Terminal: n=new game d=done 52=dice q=quit")
    if open_browser:
        webbrowser.open(url)
    threading.Thread(target=terminal, args=(game,), daemon=True).start()
    try:
        game.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.shutdown()
        if game.b is not None:
            try:
                game.b.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()


__all__ = ["main", "DEFAULTS"]
