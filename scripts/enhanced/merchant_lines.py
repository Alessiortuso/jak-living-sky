# Jak Living Sky: the white eco merchant's voice lines. One table here: each line's group (when he says it), its file
# (custom_assets/jak1/audio/voice/<file>.mp3), its subtitle id and its subtitles (Italian, English). From it:
# - the subtitles into game_custom_text_it-IT.json and en-US.json;
# - the block of collectables.gc between its markers: each line's (subtitle, length) made once, and merchant-group-say
#   (a group's lines: "first" all in order, the others one at random). Lengths from the files (ffprobe).
#
#   python scripts/enhanced/merchant_lines.py
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VOICE = ROOT / 'custom_assets/jak1/audio/voice'

LINES = [  # group, file, id, it, en
    ('first', 'merchant-first-1', 0x4664, "OH, UN CLIENTE. O UN LADRO. CON QUELLA FACCIA È DIFFICILE DIRLO.",
     "OH, A CUSTOMER. OR A THIEF. WITH THAT FACE IT'S HARD TO TELL."),
    ('first', 'merchant-first-2', 0x4665, "ECO BIANCO, EH? LO SENTO DA QUI. AVVICINATI, NON MORDO... DI SOLITO.",
     "WHITE ECO, EH? I CAN SMELL IT FROM HERE. COME CLOSER, I DON'T BITE... USUALLY."),
    ('hello', 'merchant-hello-1', 0x4666, "DI NUOVO TU. LE MIE MERCI TI MANCAVANO, EH... O ERO IO?",
     "YOU AGAIN. MISSED MY WARES, EH... OR WAS IT ME?"),
    ('hello', 'merchant-hello-4', 0x4667, "IL TOPO ARANCIONE PUÒ RESTARE. MA NON TOCCA NIENTE.",
     "THE ORANGE RAT CAN STAY. BUT HE TOUCHES NOTHING."),
    ('open', 'merchant-open-1', 0x4668, "GUARDA PURE. GUARDARE È GRATIS. PER ORA.",
     "GO ON, LOOK. LOOKING IS FREE. FOR NOW."),
    ('open', 'merchant-open-3', 0x4669, "SCEGLI CON CALMA. IO HO TUTTO IL TEMPO DEL MONDO. TU, FORSE, NO.",
     "TAKE YOUR TIME. I HAVE ALL THE TIME IN THE WORLD. YOU, MAYBE, DON'T."),
    ('buy', 'merchant-buy-2', 0x466a, "OTTIMA SCELTA. LO DICO A TUTTI, MA CON TE LO PENSO DAVVERO.",
     "GREAT CHOICE. I SAY IT TO EVERYONE, BUT WITH YOU I MEAN IT."),
    ('buy', 'merchant-buy-1', 0x466b, "AFFARE FATTO. PER ME, ALMENO.",
     "DEAL. FOR ME, AT LEAST."),
    ('buy-dark', 'merchant-buy-dark', 0x466c, "ECO OSCURO, EH?... LO SAPEVO CHE ERI UNO DI QUELLI... NON GIUDICO. INCASSO.",
     "DARK ECO, EH?... I KNEW YOU WERE ONE OF THOSE... I DON'T JUDGE. I CASH IN."),
    ('poor', 'merchant-poor-1', 0x466d, "CON QUELLA MISERIA? TORNA QUANDO AVRAI QUALCOSA DA OFFRIRE.",
     "WITH THAT PITTANCE? COME BACK WHEN YOU HAVE SOMETHING TO OFFER."),
    ('poor', 'merchant-poor-2', 0x466e, "IL CREDITO NON LO FACCIO. NEANCHE ALLE FACCE SIMPATICHE.",
     "I DON'T GIVE CREDIT. NOT EVEN TO NICE FACES."),
    ('max', 'merchant-max-1', 0x466f, "NE HAI GIÀ ABBASTANZA. ANCHE L'AVIDITÀ HA UN LIMITE. LA MIA NO, LA TUA SÌ.",
     "YOU HAVE ENOUGH ALREADY. EVEN GREED HAS A LIMIT. MINE DOESN'T, YOURS DOES."),
    ('bye-nothing', 'merchant-bye-nothing-2', 0x4670, "TORNERAI. TORNANO SEMPRE.",
     "YOU'LL BE BACK. THEY ALWAYS COME BACK."),
    ('bye', 'merchant-bye-1', 0x4671, "ALLA PROSSIMA. SE CI SARÀ UNA PROSSIMA.",
     "UNTIL NEXT TIME. IF THERE IS A NEXT TIME."),
    ('bye', 'merchant-bye-2', 0x4672, "VAI PURE. IO RESTO QUI. RESTO SEMPRE QUI.",
     "GO ON. I STAY HERE. I ALWAYS STAY HERE."),
    ('idle', 'merchant-idle-1', 0x4673, "MI STAI FISSANDO. COSTA ANCHE QUELLO.",
     "YOU'RE STARING AT ME. THAT COSTS TOO."),
    ('hit', 'merchant-hit-2', 0x4674, "RIPROVACI E I PREZZI RADDOPPIANO.",
     "TRY THAT AGAIN AND PRICES DOUBLE."),
]

BEGIN = ';; ==merchant lines (scripts/enhanced/merchant_lines.py: edit there)=='
END = ';; ==end merchant lines=='


def duration(f):
    out = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', str(f)],
                         capture_output=True, text=True).stdout
    return float(out.strip())


def put_texts(path, lang):
    p = ROOT / path
    raw = p.read_bytes().decode('utf-8')
    d = json.loads(raw)
    for g, f, i, it, en in LINES:
        d['%04x' % i] = it if lang == 'it' else en
    nl = '\r\n' if '\r\n' in raw else '\n'
    p.write_bytes((json.dumps(d, ensure_ascii=False, indent=2).replace('\n', nl) + nl).encode('utf-8'))


put_texts('game/assets/jak1/text/game_custom_text_it-IT.json', 'it')
put_texts('game/assets/jak1/text/game_custom_text_en-US.json', 'en')

out = [BEGIN]
groups = {}
for g, f, i, it, en in LINES:
    ticks = int(round((duration(VOICE / (f + '.mp3')) + 0.1) * 300))
    out.append("(define *mline-%s* (new 'static 'boxed-array :type int32 #x%x %d))" % (f[len('merchant-'):], i, ticks))
    groups.setdefault(g, []).append(f)
out.append(';; a group\'s lines said ("first": all, in order; the others: one at random)')
out.append('(defun merchant-group-say ((group symbol))')
out.append('  (case group')
for g, fs in groups.items():
    def say(f):
        return '(merchant-say "voice/%s.mp3" *mline-%s*)' % (f, f[len('merchant-'):])
    if g == 'first':
        out.append("    (('%s)" % g)
        for f in fs:
            out.append('     ' + say(f))
        out.append('     )')
    else:
        out.append("    (('%s)" % g)
        out.append('     (case (rand-vu-int-count %d)' % len(fs))
        for k, f in enumerate(fs):
            out.append('       ((%d) %s)' % (k, say(f)) if k < len(fs) - 1 else '       (else %s))' % say(f))
        out.append('     )')
out.append('    )')
out.append('  (none))')
out.append(END)
block = '\n'.join(out)

p = ROOT / 'goal_src/jak1/engine/common-obs/collectables.gc'
raw = p.read_bytes().decode('utf-8')
nl = '\r\n' if '\r\n' in raw else '\n'
t = raw.replace('\r\n', '\n')
a = t.index(BEGIN)
b = t.index(END) + len(END)
t = t[:a] + block + t[b:]
p.write_bytes(t.replace('\n', nl).encode('utf-8'))
print('ok', len(LINES), 'lines', {g: len(fs) for g, fs in groups.items()})
