"""Remote play (Nadav 16.11): a person far away plays the robot's side from a phone.

The game page's Settings -> Mode = remote. Then:
  - a second small web server starts, on this PC only (127.0.0.1:8766, nobody on the WiFi
    can reach it), with one page: the far player's board, their dice, tap a checker and
    where it goes, or Suggest (gnubg's move). Nothing else of the game is reachable there.
  - a Cloudflare quick tunnel (cloudflared, no account, downloaded once to pc/tools/) gives it
    a public https address. The link = that address + a random key; without the key every
    request gets "not found". A new link every time game.py starts; it dies when game.py closes.
  - the game page shows the link with a Copy button, and whether the far player is online.

Their side is the game's 'robot' side: the robot drags their checkers; you play yours by hand.
"""
import json
import os
import platform
import re
import secrets
import shutil
import subprocess
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import rules

PORT = 8766
TOOLS = Path(__file__).resolve().parent.parent / "tools"
ONLINE_S = 6          # the far player counts as online if their phone asked within this many seconds
NAME_MAX = 16


def cloudflared_url():
    sysname, arch = platform.system().lower(), platform.machine().lower()
    a = "arm64" if arch in ("arm64", "aarch64") else "amd64"
    base = "https://github.com/cloudflare/cloudflared/releases/latest/download/"
    if sysname == "windows":
        return base + "cloudflared-windows-amd64.exe"
    if sysname == "darwin":
        return base + f"cloudflared-darwin-{a}.tgz"
    return base + f"cloudflared-linux-{a}"


class Remote:
    def __init__(self, game, port=PORT, log=print):
        self.game, self.port, self.log = game, port, log
        self.key = secrets.token_urlsafe(9)
        self.url = None             # the public address, once the tunnel is up
        self.state = "off"          # off | downloading | starting | on | failed
        self.error = ""
        self.seen = 0.0             # when the far player's phone last asked
        self.proc = None
        self.lock = threading.Lock()
        self.srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(self))
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    # -- the link -------------------------------------------------------------
    def link(self):
        return f"{self.url}/?k={self.key}" if self.url else None

    def status(self):
        return {"state": self.state, "link": self.link(), "error": self.error,
                "online": time.time() - self.seen < ONLINE_S, "name": self.game.remote_name,
                "local": f"http://127.0.0.1:{self.port}/?k={self.key}"}

    def start_link(self):
        with self.lock:
            if self.state in ("downloading", "starting", "on"):
                return
            self.state, self.error = "starting", ""
        threading.Thread(target=self._tunnel, daemon=True).start()

    def _exe(self):
        name = "cloudflared.exe" if os.name == "nt" else "cloudflared"
        local = TOOLS / name
        if local.exists():
            return str(local)
        found = shutil.which("cloudflared")
        if found:
            return found
        self.state = "downloading"
        self.log("remote: downloading cloudflared (once)…")
        TOOLS.mkdir(parents=True, exist_ok=True)
        tmp = local.with_suffix(".part")
        url = cloudflared_url()
        with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        if url.endswith(".tgz"):
            import tarfile
            with tarfile.open(tmp) as t:
                t.extract("cloudflared", TOOLS)
            tmp.unlink()
        else:
            tmp.replace(local)
        if os.name != "nt":
            local.chmod(0o755)
        return str(local)

    def _tunnel(self):
        try:
            exe = self._exe()
            self.state = "starting"
            flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            self.proc = subprocess.Popen(
                [exe, "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{self.port}"],
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, errors="replace",
                creationflags=flags)
            import atexit
            atexit.register(self.stop)
            for line in self.proc.stderr:
                m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
                if m and not self.url:
                    self.url, self.state = m.group(0), "on"
                    self.log(f"remote link: {self.link()}")
            self.state, self.url = "failed", None
            self.error = "the link closed"
        except Exception as e:
            self.state, self.error = "failed", str(e)[:120]
            self.log(f"remote: {e}")

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()

    # -- what the far player's phone sees and does --------------------------------
    def snapshot(self):
        g = self.game
        self.seen = time.time()
        with g.lock:
            phase = g.phase
            mine = g.turn == "robot"
            if g.winner:
                what = "won" if g.winner["who"] == "robot" else "lost"
            elif phase in ("idle", "setup", "error"):
                what = "wait"
            elif phase == "remote_roll" and "remote_roll" in g.asks:
                what = "roll"
            elif phase == "remote_roll":
                what = "rolled"        # the dice are thrown; gnubg looks for its suggestion
            elif phase == "remote_move":
                what = "move"
            elif mine:
                what = "robot"         # the robot drags their checkers (or works out the dice)
            else:
                what = "host"          # the person at the board plays
            return {"what": what, "board": g.board, "dice": g.dice, "turn": g.turn,
                    "me": g.remote_name, "host": g.host_name, "white": g.robot_white,
                    "plays": [{"t": t, "s": [[x.frm, x.to] for x in s]} for t, s in g.remote_plays]
                    if what == "move" else [],
                    "hint": g.remote_hint if what == "move" else None,
                    "history": [{"who": h["who"], "dice": h["dice"], "move": h["move"], "note": h["note"]}
                                for h in g.history[-6:]],
                    "off": list(rules.off_count(g.board)), "pips": list(rules.pips(g.board)),
                    "winner": g.winner, "photo_t": getattr(g, "photo_t", 0),
                    "has_photo": g.photo_top is not None}

    def act(self, a):
        g = self.game
        name = a.get("name")
        if name == "name":
            n = re.sub(r"[^\w .'-]", "", str(a.get("value", "")), flags=re.U).strip()[:NAME_MAX]
            if n:
                g.remote_name = n
        elif name == "roll" and g.phase == "remote_roll":
            g.post({"name": "remote_roll"})
        elif name == "move" and g.phase == "remote_move":
            g.post({"name": "remote_move", "value": str(a.get("value", ""))[:80]})


def make_handler(rem):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body, ctype, code=200):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Robots-Tag", "noindex")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            self.wfile.write(body)

        def _ok(self):
            q = self.path.split("?", 1)[1] if "?" in self.path else ""
            keys = [p[2:] for p in q.split("&") if p.startswith("k=")]
            if keys and secrets.compare_digest(keys[0], rem.key):
                return True
            self._send(b"not found", "text/plain", 404)
            return False

        def do_GET(self):
            path = self.path.split("?")[0]
            if not self._ok():
                return
            if path == "/":
                self._send(PAGE.encode(), "text/html; charset=utf-8")
            elif path == "/s":
                self._send(json.dumps(rem.snapshot()).encode(), "application/json")
            elif path == "/board.jpg" and rem.game.photo_top:
                self._send(rem.game.photo_top, "image/jpeg")
            else:
                self._send(b"not found", "text/plain", 404)

        def do_POST(self):
            if self.path.split("?")[0] != "/a" or not self._ok():
                return
            n = min(int(self.headers.get("Content-Length", 0)), 2000)
            try:
                rem.act(json.loads(self.rfile.read(n) or b"{}"))
            except (ValueError, AttributeError):
                return self._send(b"bad", "text/plain", 400)
            self._send(b"{}", "application/json")
    return H


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<meta name="robots" content="noindex"><meta name="color-scheme" content="dark only"><meta name="theme-color" content="#12100e">
<title>Backgammon with the robot</title>
<style>
:root{color-scheme:only dark;--bg:#12100e;--card:#1d1915;--ink:#f3ebdd;--mute:#a59882;--line:#3a3128;--gold:#e2b053}
svg,img{forced-color-adjust:none}
*{box-sizing:border-box;-webkit-tap-highlight-color:transparent}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.35 system-ui,-apple-system,"Segoe UI",sans-serif;min-height:100vh}
main{max-width:560px;margin:0 auto;padding:12px 12px 28px;display:grid;gap:10px}
.top{display:flex;justify-content:space-between;align-items:center;gap:8px;font-weight:700}
.who{display:flex;align-items:center;gap:8px;padding:6px 12px;border-radius:30px;background:linear-gradient(135deg,#d9b57c,#9a7646);font-size:15px}
.who.on{box-shadow:0 0 0 3px var(--gold),0 0 20px #e2b05388}
.who i{width:20px;height:20px;border-radius:50%;display:inline-block;border:2px solid #777}
.vs{color:var(--mute);font-size:13px}
#prompt{font-size:22px;font-weight:800;text-align:center;min-height:30px}
#sub{text-align:center;color:var(--mute);font-size:14px;min-height:19px}
svg{width:100%;height:auto;display:block;border-radius:12px;touch-action:manipulation}
.row{display:flex;gap:10px;justify-content:center;align-items:center;flex-wrap:wrap}
button,select,input{font:inherit;border-radius:12px;border:1px solid var(--line);padding:10px 16px;background:var(--card);color:var(--ink)}
button{background:linear-gradient(180deg,#f0c56d,#d79f3c);color:#231a0c;border:0;font-weight:800;box-shadow:0 2px 0 #8a6220}
button:active{transform:translateY(2px)}
button.sec{background:transparent;color:var(--ink);border:1px solid var(--line);box-shadow:none;font-weight:600}
button.big{font-size:24px;padding:14px 54px}
button[disabled]{opacity:.4}
.dice{display:flex;gap:16px;justify-content:center;min-height:76px}
.die{width:72px;height:72px;filter:drop-shadow(0 4px 6px #000a)}
.die.rolling{animation:tumble .3s linear infinite}
@keyframes tumble{to{transform:rotate(360deg)}}
.log{font-size:13px;color:var(--mute);display:grid;gap:2px}
details{color:var(--mute);font-size:14px}
#cam{width:100%;border-radius:10px;margin-top:8px}
.spin{display:inline-block;width:18px;height:18px;border-radius:50%;border:3px solid #ffffff22;border-top-color:var(--gold);animation:tumble .8s linear infinite;vertical-align:-3px;margin-left:8px}
#namebox{display:none;gap:8px;justify-content:center}
</style></head><body><main>
<div class="top"><div class="who" id="w-me"><i id="c-me"></i><span id="n-me">You</span></div><span class="vs">vs</span>
 <div class="who" id="w-host"><span id="n-host">Nadav</span><i id="c-host"></i></div></div>
<div class="row" id="namebox"><input id="name" maxlength="16" placeholder="Your name" size="12"><button class="sec" id="namebtn">OK</button></div>
<div id="prompt">Connecting…</div><div id="sub"></div>
<svg id="b" viewBox="0 0 400 340"></svg>
<div class="dice" id="dice"></div>
<div class="row" id="acts"></div>
<div class="log" id="log"></div>
<details><summary>The real board (camera from below)</summary><img id="cam" alt="the real board"></details>
</main><script>
const K=new URLSearchParams(location.search).get("k");
let ROLLED=false,S=null,SEL=null,CH=[],lastKey="",lastWhat="",lastPhoto=0,ROLLING=0,AC=null;
const $=id=>document.getElementById(id);
function post(a){return fetch("/a?k="+encodeURIComponent(K),{method:"POST",body:JSON.stringify(a)})}
// -- sound and buzz: your turn --
function beep(){try{if(!AC)return;const t=AC.currentTime;for(const [f,d] of [[660,0],[880,.12]]){const o=AC.createOscillator(),g=AC.createGain();
 o.frequency.value=f;g.gain.setValueAtTime(.0001,t+d);g.gain.exponentialRampToValueAtTime(.25,t+d+.02);g.gain.exponentialRampToValueAtTime(.0001,t+d+.25);
 o.connect(g).connect(AC.destination);o.start(t+d);o.stop(t+d+.3)}}catch(e){}if(navigator.vibrate)navigator.vibrate([80,60,80])}
document.addEventListener("pointerdown",()=>{if(!AC){try{AC=new (window.AudioContext||window.webkitAudioContext)()}catch(e){}}},{once:true});
// -- the board, from the far player's side: their home bottom right (1-6), 13-24 along the top --
const W=400,H=340,BX=188,BW=24,PW=28,TOP=10,BOT=330,PH=130,OFFX=372;
function px(p){// x of the middle of point p (1..24)
 const i=p<=12?12-p:p-13;return 10+i*PW+PW/2+(i>=6?BW:0)}
function up(p){return p>=13}
function slotY(p,k){const r=13;return up(p)?TOP+r+k*2*r:BOT-r-k*2*r}
// pure white and pure black, each with a rim in the other colour, so they read on any point and any screen
function col(mine){return (mine===!!S.white)?["#ffffff","#111111"]:["#111111","#e8e8e8"]}
function disc(x,y,mine,hl=0){const [f,s]=col(mine);// hl: 1 = can move, 2 = picked (a yellow rim)
 return `<circle cx="${x}" cy="${y}" r="12.4" fill="${f}" stroke="${hl?"#ffd400":s}" stroke-width="${hl===2?4.5:hl?3:2}"/>`+
 `<circle cx="${x}" cy="${y}" r="7" fill="none" stroke="${s}" stroke-width="1" opacity=".55" pointer-events="none"/>`}
function shown(){// the board with the steps chosen so far
 const b=S.board.slice();for(const [f,t] of CH){if(f===25)b[0]--;else b[f]--;if(t>0){if(b[t]===-1){b[t]=0;b[25]--}b[t]++}}return b}
function remaining(){// the plays that still fit the steps chosen so far, and what is left of each
 const out=[];for(const p of S.plays){const left=p.s.map(x=>x.slice());let ok=true;
  for(const [f,t] of CH){const i=left.findIndex(x=>x[0]===f&&x[1]===t);if(i<0){ok=false;break}left.splice(i,1)}
  if(ok)out.push({p,left})}return out}
function targets(from){// where a checker from `from` can go now: one die, or more in a row
 const t=new Map();for(const {left} of remaining()){
  const walk=(at,rest,path)=>{for(let i=0;i<rest.length;i++){if(rest[i][0]!==at)continue;
   const np=path.concat([rest[i]]),to=rest[i][1];if(!t.has(to)||t.get(to).length>np.length)t.set(to,np);
   if(to>0)walk(to,rest.slice(0,i).concat(rest.slice(i+1)),np)}};walk(from,left,[])}
 return t}
function sources(){const s=new Set();for(const {left} of remaining())for(const [f] of left)s.add(f);return s}
function complete(){return remaining().find(r=>r.left.length===0)}
function draw(){const b=S.board?shown():Array(26).fill(0),mv=S.what==="move";
 const src=mv?sources():new Set(),tg=mv&&SEL!==null?targets(SEL):new Map();
 let h=`<rect width="${W}" height="${H}" fill="#4a2f1c"/><rect x="6" y="6" width="${W-12-34}" height="${H-12}" rx="6" fill="#1d4d3f"/>`;
 h+=`<rect x="${BX}" y="6" width="${BW}" height="${H-12}" fill="#3a2414"/>`;
 h+=`<rect x="${OFFX-12}" y="6" width="38" height="${H-12}" rx="5" fill="#2a1a0e"/>`;
 for(let p=1;p<=24;p++){const x=px(p),u=up(p),y0=u?TOP:BOT,y1=u?TOP+PH:BOT-PH;
  h+=`<polygon points="${x-PW/2+1},${y0} ${x+PW/2-1},${y0} ${x},${y1}" fill="${p%2?'#9c7a4a':'#7a2a1c'}"/>`;
  h+=`<text x="${x}" y="${u?TOP+PH+14:BOT-PH-6}" font-size="10" fill="#cdbf9f" text-anchor="middle">${p}</text>`;
  if(tg.has(p))h+=`<polygon points="${x-PW/2+1},${y0} ${x+PW/2-1},${y0} ${x},${y1}" fill="#ffe14d" opacity=".45"/>`}
 for(let p=1;p<=24;p++){const n=Math.abs(b[p]);if(!n)continue;const mine=b[p]>0,x=px(p);
  for(let k=0;k<Math.min(n,5);k++){const sel=mine&&SEL===p&&k===Math.min(n,5)-1;
   h+=disc(x,slotY(p,k),mine,sel?2:(mine&&src.has(p)&&k===Math.min(n,5)-1&&SEL===null?1:0))}
  if(n>5)h+=`<text x="${x}" y="${slotY(p,4)+5}" font-size="14" font-weight="bold" text-anchor="middle" fill="${col(mine)[0]==='#f2f2ea'?'#000':'#fff'}">${n}</text>`}
 // the bar: theirs at the bottom half, the opponent's at the top
 for(let k=0;k<b[0];k++)h+=disc(BX+BW/2,H/2+20+k*26,true,SEL===25?2:(src.has(25)?1:0));
 for(let k=0;k<-b[25];k++)h+=disc(BX+BW/2,H/2-20-k*26,false);
 // borne off: theirs bottom, the opponent's top
 const off=15-b[0]-b.slice(1,25).reduce((a,v)=>a+(v>0?v:0),0),oo=15+b[25]-b.slice(1,25).reduce((a,v)=>a+(v<0?-v:0),0);
 const [mf]=col(true),[of]=col(false);
 for(let k=0;k<off;k++)h+=`<rect x="${OFFX-8}" y="${BOT-8-k*9}" width="30" height="7" rx="2" fill="${mf}"/>`;
 for(let k=0;k<oo;k++)h+=`<rect x="${OFFX-8}" y="${TOP+1+k*9}" width="30" height="7" rx="2" fill="${of}"/>`;
 if(tg.has(0))h+=`<rect x="${OFFX-12}" y="${H/2}" width="38" height="${H/2-6}" rx="5" fill="#ffe14d" opacity=".45"/><text x="${OFFX+7}" y="${H/2+20}" font-size="12" text-anchor="middle" fill="#fff">off</text>`;
 $("b").innerHTML=h}
function hitTest(ev){const r=$("b").getBoundingClientRect(),x=(ev.clientX-r.left)*W/r.width,y=(ev.clientY-r.top)*H/r.height;
 if(x>=OFFX-12)return 0;if(x>=BX&&x<BX+BW)return 25;
 let best=null,d=1e9;for(let p=1;p<=24;p++){const dx=Math.abs(px(p)-x);if(dx<PW/2&&(up(p)===(y<H/2))){if(dx<d){d=dx;best=p}}}return best}
$("b").addEventListener("click",ev=>{if(!S||S.what!=="move")return;const p=hitTest(ev);if(p===null)return;
 if(SEL!==null){const tg=targets(SEL);if(tg.has(p)){CH=CH.concat(tg.get(p));SEL=null;after();return}}
 SEL=(sources().has(p)&&p!==0)?(SEL===p?null:p):null;draw()});
function after(){const c=complete();draw();acts();if(c)$("sub").textContent="Your move: "+c.p.t}
// -- dice --
function face(v){const P={1:[[50,50]],2:[[28,28],[72,72]],3:[[28,28],[50,50],[72,72]],4:[[28,28],[72,28],[28,72],[72,72]],
 5:[[28,28],[72,28],[50,50],[28,72],[72,72]],6:[[28,26],[72,26],[28,50],[72,50],[28,74],[72,74]]}[v]||[];
 return `<svg viewBox="0 0 100 100" class="die${Date.now()<ROLLING?' rolling':''}"><rect x="4" y="4" width="92" height="92" rx="18" fill="#ffffff" stroke="#e2b053" stroke-width="5"/>`+
  P.map(([x,y])=>`<circle cx="${x}" cy="${y}" r="11" fill="#000000"/>`).join("")+`</svg>`}
function dice(){const d=S.dice;$("dice").innerHTML=d?d.map(face).join(""):""}
// -- buttons --
function el(t,a={}){const e=document.createElement(t);for(const k in a){if(k==="text")e.textContent=a[k];else if(k.startsWith("on"))e[k]=a[k];else e.setAttribute(k,a[k])}return e}
function acts(){const a=$("acts");a.innerHTML="";const w=S.what;
 if(w==="roll")a.append(el("button",{class:"big",text:"Roll",onclick:()=>{ROLLING=Date.now()+800;post({name:"roll"});$("acts").innerHTML=""}}));
 if(w==="move"){const c=complete();
  a.append(el("button",{class:c?"big":"",text:c?"Play":"Pick checkers",...(c?{}:{disabled:""}),onclick:()=>{if(c){post({name:"move",value:c.p.t});CH=[];SEL=null;$("acts").innerHTML="";$("sub").textContent="Sending…"}}}));
  if(CH.length)a.append(el("button",{class:"sec",text:"Undo",onclick:()=>{CH=[];SEL=null;$("sub").textContent="";draw();acts()}}));
  if(S.hint)a.append(el("button",{class:"sec",text:"Suggest",onclick:()=>{const h=S.plays.find(p=>p.t===S.hint);if(h){CH=h.s.map(x=>x.slice());SEL=null;after()}}}));
  const sel=el("select");sel.append(el("option",{text:"or pick from the list…",value:""}));for(const p of S.plays)sel.append(el("option",{text:p.t,value:p.t}));
  sel.onchange=()=>{const h=S.plays.find(p=>p.t===sel.value);if(h){CH=h.s.map(x=>x.slice());SEL=null;after()}};a.append(sel)}}
const SAY={wait:["Waiting for the game to start","%h sets up the board"],roll:["Your turn!","Tap Roll"],
 rolled:["Rolling…",""],move:["Your move","Tap a checker, then where it goes"],robot:["The robot is moving your checkers…",""],
 host:["%h is playing…","The camera reads the move"],won:["You win! 🎉",""],lost:["%h wins",""]};
function render(s){const first=!S;S=s;const w=s.what;
 $("n-me").textContent=s.me==="Guest"?"You":s.me;$("n-host").textContent=s.host;
 $("c-me").style.background=col(true)[0];$("c-host").style.background=col(false)[0];
 $("w-me").classList.toggle("on",s.turn==="robot"&&!s.winner);$("w-host").classList.toggle("on",s.turn==="human"&&!s.winner);
 if(first&&s.me==="Guest")$("namebox").style.display="flex";
 const k=JSON.stringify([w,s.board,s.dice,s.plays.length]);
 if(k!==lastKey){lastKey=k;if(w!=="move"){CH=[];SEL=null}
  const [p,sub]=SAY[w]||["",""];$("prompt").innerHTML=p.replace("%h",s.host)+(w==="robot"||w==="host"||w==="rolled"?'<span class="spin"></span>':"");
  $("sub").textContent=sub.replace("%h",s.host);
  if((w==="roll")&&lastWhat!=="roll")beep();lastWhat=w;dice();draw();acts();
  $("log").innerHTML="";for(const m of s.history.slice().reverse()){const d=el("div");
   d.textContent=(m.who==="robot"?"You":s.host)+" "+m.dice.join("-")+": "+(m.note==="no legal move"?"no legal move":m.move);$("log").append(d)}}
 if(Date.now()<ROLLING||ROLLED){ROLLED=Date.now()<ROLLING;dice()}
 if(s.has_photo&&s.photo_t!==lastPhoto){lastPhoto=s.photo_t;$("cam").src="/board.jpg?k="+encodeURIComponent(K)+"&t="+s.photo_t}}
$("namebtn").onclick=()=>{const v=$("name").value.trim();if(v){post({name:"name",value:v});$("namebox").style.display="none"}};
async function tick(){try{render(await (await fetch("/s?k="+encodeURIComponent(K))).json())}catch(e){$("prompt").textContent="Connecting…"}setTimeout(tick,700)}
tick();
</script></body></html>"""
