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
.dice{display:inline-flex;gap:8px;margin-left:10px;vertical-align:middle}.die{display:inline-block;width:38px;height:38px}
.die.rolling{animation:tumble .35s linear infinite}
@keyframes tumble{0%{transform:rotate(0) translateY(0)}25%{transform:rotate(90deg) translateY(-6px)}
 50%{transform:rotate(180deg) translateY(0)}75%{transform:rotate(270deg) translateY(-4px)}100%{transform:rotate(360deg)}}
svg{width:100%;height:auto;display:block}
.warn{color:var(--warn);font-size:13px}
table{width:100%;border-collapse:collapse;font-size:13px}td{padding:3px 4px;border-bottom:1px solid var(--line)}
#photo,.photo2{width:100%;border-radius:8px;display:none;margin-top:8px}
.grid{display:grid;grid-template-columns:auto 1fr;gap:6px 10px;align-items:center}
.stats{display:flex;flex-wrap:wrap;gap:4px 16px;font-size:13px;margin-top:6px}
</style></head><body><main>
<section class="card">
  <h1>AI-bg <span class="mute" id="sim"></span></h1>
  <div id="prompt">…</div>
  <div id="asks"></div>
  <div class="stats"><b id="turn"></b><span id="pips"></span><span id="offs"></span></div>
  <div id="coach" class="mute"></div>
  <div id="warns"></div>
  <svg id="board" viewBox="-22 -22 454 464"></svg>
  <div class="mute">Drawn the same way round as the camera's top view next to it. Numbers are your columns.
  White = robot, red = you.</div>
</section>
<section style="display:grid;gap:16px;align-content:start">
  <div class="card"><h1>What the camera sees</h1><div class="mute">The board from above, circles = checkers found.</div>
   <img id="photo" alt="top view">
   <details style="margin-top:8px"><summary class="mute">Original camera photo</summary>
    <img id="top" class="photo2" alt="raw photo"></details></div>
  <details class="card"><summary class="mute">Settings (used from the next New game)</summary>
   <div class="grid" id="settings" style="margin-top:8px"></div></details>
  <table id="hist" hidden></table>
</section></main>
<script>
const OPTS={mode:["robot_vs_human","robot_vs_robot","watch"],engine:["gnubg","simple","random"],
 dice:["digital","typed","camera"],human_move:["done_button","auto","robot_moves"],
 vision:["learned","norm"],coach:[false,true],first:["roll","robot","human"],board:["auto","real","sim"]};
const NAMES={mode:"Mode",engine:"Engine",dice:"Dice",human_move:"Your move",vision:"Vision",
 coach:"Coach",first:"Who starts",board:"Board"};
const BTN={new:"New game",done:"Done",accept:"Accept",reread:"Read again",closest:"Take closest legal move",
 learn:"Learn colours",arrange:"Robot arranges the board",rolled:"Rolled"};
let S=null, built=false, lastPhoto=0, ED=null, EDC="robot";
// -- the board is drawn as the corrected camera photo shows it: frame mm (x, y from the near side)
// -> magnet mm (x right, y down) through the layout's own map (s.frame), 1 unit = 1 mm --
function P(x,y){const A=S.frame;return [A[0][0]*x+A[0][1]*y+A[0][2],A[1][0]*x+A[1][1]*y+A[1][2]]}
function Q(x,y){const [a,b]=P(x,y);return `${a.toFixed(1)},${b.toFixed(1)}`}
function toFrame(mx,my){const A=S.frame,d=A[0][0]*A[1][1]-A[0][1]*A[1][0],u=mx-A[0][2],v=my-A[1][2];
 return [(A[1][1]*u-A[0][1]*v)/d,(-A[1][0]*u+A[0][0]*v)/d]}
function txt(x,y,t,size,fill,bold){const [a,b]=P(x,y);
 return `<text x="${a}" y="${b+size*0.35}" font-size="${size}" ${bold?'font-weight="bold"':''} text-anchor="middle" fill="${fill}">${t}</text>`}
function disc(x,y,side){const [a,b]=P(x,y);return `<circle cx="${a}" cy="${b}" r="12.5" fill="var(--${side})" stroke="#0008" stroke-width="1.2"/>`}
function base(s,edit){let h=`<rect x="-20" y="-20" width="450" height="460" rx="10" fill="var(--card)" stroke="${edit?'#e0a000':'var(--line)'}" stroke-width="${edit?4:1}"/>`;
 if(edit)h+=`<rect x="-20" y="-20" width="450" height="460" rx="10" fill="#e0a000" opacity=".10"/>`;
 const by=s.bar_y;h+=`<polygon points="${Q(0,by-13)} ${Q(410,by-13)} ${Q(410,by+13)} ${Q(0,by+13)}" fill="var(--line)" opacity=".35"/>`;
 for(const [x,y,n] of s.labels){const left=x<205;
  const pts=left?`${Q(x-13,y+13)} ${Q(x-13,y-13)} ${Q(x+150,y)}`:`${Q(x+13,y+13)} ${Q(x+13,y-13)} ${Q(x-150,y)}`;
  h+=`<polygon points="${pts}" fill="${n%2?'#c9a77c':'#8f6a43'}" opacity=".45"/>`;
  h+=txt(left?x+162:x-162,y,n,11,"var(--mute)")}
 return h}
function edPlace(ev){if(!ED)return;const svg=document.getElementById("board");const pt=svg.createSVGPoint();
 pt.x=ev.clientX;pt.y=ev.clientY;const q=pt.matrixTransform(svg.getScreenCTM().inverse());
 const [fx,y]=toFrame(q.x,q.y),left=fx<205;
 let best=null;for(const [x,ly,n] of S.labels){if((x<205)!==left)continue;const d=Math.abs(ly-y);if(!best||d<best[0])best=[d,25-n]}
 if(Math.abs(S.bar_y-y)<(best?best[0]:1e9))best=[0,left?0:25];
 if(!best)return;const p=best[1],sign=EDC==="robot"?1:-1;
 if(p===0||p===25){if(EDC==="remove"){if(p===0&&ED[0]>0)ED[0]--;if(p===25&&ED[25]<0)ED[25]++}
   else if(p===0&&EDC==="robot")ED[0]++;else if(p===25&&EDC==="human")ED[25]--}
 else if(EDC==="remove"){if(ED[p]>0)ED[p]--;else if(ED[p]<0)ED[p]++}
 else if(ED[p]*sign<0)ED[p]=sign;else ED[p]+=sign;
 drawEdit(S)}
function offOf(b){let r=0,h=0;for(const v of b){if(v>0)r+=v;else h-=v}return [15-r,15-h]}
function stack(x,y,dir,v,side){// checkers side by side from the edge toward the middle, like a real board
 const n=Math.abs(v);if(!n)return "";const step=n>1?Math.min(26,138/(n-1)):0;let h="";
 for(let i=0;i<n;i++)h+=disc(x+dir*i*step,y,side);
 if(n>6)h+=txt(x+dir*(n-1)*step,y,n,13,ink(side),true);
 return h}
function drawEdit(s){const svg=document.getElementById("board");
 let h=base(s,true)+txt(205,s.labels[11][1],"EDITING",13,"#c08000",true);
 for(const [x,y,n] of s.labels){const v=ED[25-n];if(v)h+=stack(x,y,x<205?1:-1,v,v>0?"robot":"human")}
 for(const [x,v,side,dir] of [[60,ED[0],"robot",1],[350,-ED[25],"human",-1]]){h+=txt(x,s.bar_y,"bar",11,"var(--mute)");
  if(v)h+=stack(x+30*dir,s.bar_y,dir,v,side)}
 const o=offOf(ED);h+=txt(205,35,`off: robot ${o[0]} · you ${o[1]}`,12,"var(--mute)");
 svg.innerHTML=h}
// -- dice: pips, a short tumble and a rattle when new dice appear --
const PIPS={1:[[2,2]],2:[[1,1],[3,3]],3:[[1,1],[2,2],[3,3]],4:[[1,1],[3,1],[1,3],[3,3]],
 5:[[1,1],[3,1],[2,2],[1,3],[3,3]],6:[[1,1],[3,1],[1,2],[3,2],[1,3],[3,3]]};
function face(v){return `<svg viewBox="0 0 40 40" width="38" height="38"><rect x="1" y="1" width="38" height="38" rx="8" fill="#fff" stroke="#999"/>`+
 (PIPS[v]||[]).map(([i,j])=>`<circle cx="${i*10}" cy="${j*10}" r="3.6" fill="#222"/>`).join("")+`</svg>`}
let lastDice, rollEnd=0, AC=null;
function rolling(){return Date.now()<rollEnd}
function rattle(){try{AC=AC||new (window.AudioContext||window.webkitAudioContext)();const t0=AC.currentTime;
 for(let k=0;k<7;k++){const t=t0+k*0.07+Math.random()*0.03,n=AC.createBufferSource(),b=AC.createBuffer(1,AC.sampleRate*0.03,AC.sampleRate),
  d=b.getChannelData(0);for(let i=0;i<d.length;i++)d[i]=(Math.random()*2-1)*Math.pow(1-i/d.length,3);
  const f=AC.createBiquadFilter();f.type="bandpass";f.frequency.value=1800+Math.random()*1800;f.Q.value=2;
  const g=AC.createGain();g.gain.value=0.5*(1-k/9);n.buffer=b;n.connect(f).connect(g).connect(AC.destination);n.start(t)}}catch(e){}}
function roll(){rollEnd=Date.now()+700;rattle();const t=setInterval(()=>{const ds=document.querySelectorAll(".die");
 for(const e of ds){if(rolling())e.innerHTML=face(1+Math.floor(Math.random()*6));else{e.innerHTML=face(+e.dataset.v);e.classList.remove("rolling")}}
 if(!rolling())clearInterval(t)},80)}
document.addEventListener("pointerdown",()=>{try{AC=AC||new (window.AudioContext||window.webkitAudioContext)();AC.resume()}catch(e){}},{once:true});
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
function drawBoard(s){const svg=document.getElementById("board");
 let h=base(s,false)+txt(205,s.bar_y,"BAR",12,"var(--mute)")+txt(205,35,"bear-off",11,"var(--mute)");
 for(const [x,y,side] of s.pieces)h+=disc(x,y,side);
 svg.innerHTML=h}
function ink(side){const c=getComputedStyle(document.documentElement).getPropertyValue("--"+side).trim();
 const m=/^#([0-9a-f]{6})$/i.exec(c);if(!m)return side==="robot"?"#000":"#fff";const n=parseInt(m[1],16);
 return (0.299*(n>>16)+0.587*((n>>8)&255)+0.114*(n&255))>150?"#000":"#fff"}
function paintSides(c){const r=document.documentElement.style;
 if(c){r.setProperty("--robot",c.robot);r.setProperty("--human",c.human)}else{r.removeProperty("--robot");r.removeProperty("--human")}}
function render(s){S=s;paintSides(s.colours);if(!built){buildSettings(s.settings);built=true}
 document.getElementById("sim").textContent=s.sim?"(simulator)":"";
 const p=document.getElementById("prompt");p.textContent=s.prompt;
 const dk=JSON.stringify(s.dice||null);if(dk!==lastDice){if(lastDice!==undefined&&s.dice)roll();lastDice=dk}
 if(s.dice){const d=el("span",{class:"dice"});for(const v of s.dice){const e=el("span",{class:"die"+(rolling()?" rolling":"")});
  e.dataset.v=v;e.innerHTML=face(rolling()?1+Math.floor(Math.random()*6):v);d.append(e)}p.append(d)}
 const a=document.getElementById("asks");const keep=document.activeElement&&a.contains(document.activeElement);
 if(!keep||a.dataset.k!==JSON.stringify(s.asks)+s.legal.length){a.innerHTML="";a.dataset.k=JSON.stringify(s.asks)+s.legal.length;
  for(const k of s.asks){
   if(k==="dice_input"){const i=el("input",{placeholder:"dice, e.g. 5 2",size:8});
    a.append(i,el("button",{text:"OK",onclick:()=>post({name:"dice",value:i.value})}))}
   else if(k==="edit"){const sel=el("select",{onchange:()=>{EDC=sel.value}});
    for(const [v,t] of [["robot","click adds white (robot)"],["human","click adds red (you)"],["remove","click removes one"]])
     sel.append(el("option",{value:v,text:t}));sel.value=EDC;
    a.append(el("div",{class:"mute",text:"Fix the reading: click a point (or the bar) on the board."}),sel,
     el("button",{text:"Use this board",onclick:()=>post({name:"board",value:ED})}),
     el("button",{class:"sec",text:"Back to the camera's reading",onclick:()=>{ED=S.seen.slice();drawEdit(S)}}))}
   else if(k==="move_pick"){const sel=el("select");for(const m of s.legal)sel.append(el("option",{text:m}));
    a.append(sel,el("button",{text:"Play this",onclick:()=>post({name:"move",value:sel.value})}))}
   else a.append(el("button",{text:BTN[k]||k,onclick:()=>post({name:k})}))}
  a.append(el("button",{class:"sec",text:"New game",onclick:()=>post({name:"new"})}))}
 document.getElementById("turn").textContent=s.winner?"":(s.turn==="robot"?"🤖 Robot's turn":s.turn?"🙂 Your turn":"");
 const ahead=s.pips[0]===s.pips[1]?"even race":(s.pips[0]<s.pips[1]?"robot is ahead":"you are ahead");
 document.getElementById("pips").textContent=`Steps left to finish: robot ${s.pips[0]}, you ${s.pips[1]} (${ahead})`;
 document.getElementById("offs").textContent=(s.off[0]||s.off[1])?`Taken off: robot ${s.off[0]}/15, you ${s.off[1]}/15`:"";
 document.getElementById("coach").textContent=s.coach||"";
 const w=document.getElementById("warns");w.innerHTML="";for(const t of s.warnings)w.append(el("div",{class:"warn",text:"⚠ "+t}));
 if(s.asks.includes("edit")&&s.seen){if(!ED)ED=s.seen.slice();drawEdit(s)}else{ED=null;drawBoard(s)}
 const t=document.getElementById("hist");t.innerHTML="";
 for(const m of s.history.slice().reverse())t.append(el("tr",{},[el("td",{text:m.n}),el("td",{text:m.who}),
  el("td",{text:m.dice.join("-")}),el("td",{text:m.move}),el("td",{class:"mute",text:m.note||""})]));
 if(s.has_photo&&s.photo_t!==lastPhoto){lastPhoto=s.photo_t;const im=document.getElementById("photo");
  im.src="/top.jpg?t="+s.photo_t;im.style.display="block";
  const tp=document.getElementById("top");tp.src="/photo.jpg?t="+s.photo_t;tp.style.display="block"}}
async function tick(){try{render(await (await fetch("/state")).json())}catch(e){}setTimeout(tick,500)}
document.getElementById("board").addEventListener("click",edPlace);
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
            elif path == "/top.jpg" and getattr(game, "photo_top", None):
                self._send(game.photo_top, "image/jpeg")
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
