# E-015 — recording sheet for the 44 never-seen terms

100 sentences are imported (`reviewed: false`): **50 for S1**, **50 for S2**. Every one of
the 44 terms appears once per speaker, plus six two-term sentences. Average length 6.5 words.

Read them once before recording. If a sentence sounds unnatural, stop and tell me: I replace it
before you spend a take on it. Do not edit the text while recording — the manifest validates
the audio against the script.

## Commands

```bash
# S1: cues 0642–0691 (50 sentences)
uv run python recordings/record.py --speaker S1 --kind R --start 642

# S2: cues 0624–0673 (50 sentences)
uv run python recordings/record.py --speaker S2 --kind R --start 624
```

`record.py` skips files that already exist. To redo one bad take, delete that wav and re-run the
same command: it records only what is missing.

After recording, validate the audio against the script:

```bash
uv run python recordings/build_manifest.py build --kind R --speaker S1 --allow-unreviewed
uv run python recordings/build_manifest.py build --kind R --speaker S2 --allow-unreviewed
```

Then tell me: I build the mel shards, add the gold to the mix at x20, fine-tune from 0.7.0
for one epoch (~2 h), and evaluate on the 47 rare-term utterances plus the clean gate.

## S1 — 50 sentences

| # | id | sentence |
|---|---|---|
| 642 | `R-S1-0642` | gli airpods si sono scaricati durante la call |
| 643 | `R-S1-0643` | abbiamo preso un gelato da baskin robbins |
| 644 | `R-S1-0644` | il codice del progetto è su bitbucket |
| 645 | `R-S1-0645` | la cioccolata cadbury è finita |
| 646 | `R-S1-0646` | ho lasciato il charger in ufficio |
| 647 | `R-S1-0647` | il checkout non accetta la mia carta |
| 648 | `R-S1-0648` | il chromecast non trova la rete |
| 649 | `R-S1-0649` | il delivery è arrivato in ritardo |
| 650 | `R-S1-0650` | il domain del sito scade a dicembre |
| 651 | `R-S1-0651` | l'aspirapolvere dyson è potentissimo |
| 652 | `R-S1-0652` | il mio exam di inglese è andato bene |
| 653 | `R-S1-0653` | salva i file nella folder condivisa |
| 654 | `R-S1-0654` | il piano di hosting scade oggi |
| 655 | `R-S1-0655` | il design è su invision |
| 656 | `R-S1-0656` | il mio ipad non si carica |
| 657 | `R-S1-0657` | il kfc vicino alla stazione è pieno |
| 658 | `R-S1-0658` | fai il logout prima di chiudere |
| 659 | `R-S1-0659` | i mondays in ufficio sono pesanti |
| 660 | `R-S1-0660` | domani ho un oral exam di inglese |
| 661 | `R-S1-0661` | chiudi quel popup e riprova |
| 662 | `R-S1-0662` | il prototype è pronto per la demo |
| 663 | `R-S1-0663` | devo fare un reset della password |
| 664 | `R-S1-0664` | fai un restart del server |
| 665 | `R-S1-0665` | il roomba è bloccato sotto il divano |
| 666 | `R-S1-0666` | mandami uno screenshot dell'errore |
| 667 | `R-S1-0667` | ci siamo fatti un selfie in ufficio |
| 668 | `R-S1-0668` | lo screen sharing non funziona |
| 669 | `R-S1-0669` | lo shutdown è previsto per le sei |
| 670 | `R-S1-0670` | le foto sono prese da shutterstock |
| 671 | `R-S1-0671` | ho aggiunto le skills nel profilo |
| 672 | `R-S1-0672` | ho cercato il volo su skyscanner |
| 673 | `R-S1-0673` | lo smartwatch mi conta i passi |
| 674 | `R-S1-0674` | facciamo una sessione di problem solving |
| 675 | `R-S1-0675` | ho preso un caffè da starbucks |
| 676 | `R-S1-0676` | la startup ha chiuso un round |
| 677 | `R-S1-0677` | ho mangiato un panino da subway |
| 678 | `R-S1-0678` | fai swipe a destra per accettare |
| 679 | `R-S1-0679` | ho letto le recensioni su tripadvisor |
| 680 | `R-S1-0680` | il tè twinings è nella credenza |
| 681 | `R-S1-0681` | ho fatto unfollow a quel profilo |
| 682 | `R-S1-0682` | il file è in upload da stamattina |
| 683 | `R-S1-0683` | ho cambiato il wallpaper del desktop |
| 684 | `R-S1-0684` | il website aziendale è lento |
| 685 | `R-S1-0685` | il checkout del website è lento |
| 686 | `R-S1-0686` | prima un reset poi un restart del router |
| 687 | `R-S1-0687` | metti lo screenshot nella folder del progetto |
| 688 | `R-S1-0688` | il file è in upload e lo screen sharing non parte |
| 689 | `R-S1-0689` | abbiamo fatto un selfie da starbucks |
| 690 | `R-S1-0690` | il mio ipad è scarico e il charger è a casa |
| 691 | `R-S1-0691` | la scaleup ha assunto tre sviluppatori |

## S2 — 50 sentences

**Recorded 2026-11-11:** 50/50 clips, 2.1 min, 48 kHz mono, builder 0 errors; the full R-S2
manifest now has 672 valid rows. The second arm is E-028 (`runs/E-028/`).

| # | id | sentence |
|---|---|---|
| 624 | `R-S2-0624` | gli airpods nuovi hanno una buona durata |
| 625 | `R-S2-0625` | il gelato di baskin robbins è troppo dolce |
| 626 | `R-S2-0626` | abbiamo spostato i repository su bitbucket |
| 627 | `R-S2-0627` | ho comprato una tavoletta cadbury |
| 628 | `R-S2-0628` | il charger del telefono è nel cassetto |
| 629 | `R-S2-0629` | il checkout online richiede la verifica |
| 630 | `R-S2-0630` | ho collegato il chromecast alla televisione |
| 631 | `R-S2-0631` | il delivery costa cinque euro |
| 632 | `R-S2-0632` | abbiamo comprato un domain nuovo |
| 633 | `R-S2-0633` | il dyson fa più rumore del previsto |
| 634 | `R-S2-0634` | il mio exam di domani è stato spostato |
| 635 | `R-S2-0635` | crea una folder per ogni cliente |
| 636 | `R-S2-0636` | abbiamo cambiato hosting per risparmiare |
| 637 | `R-S2-0637` | il prototipo è caricato su invision |
| 638 | `R-S2-0638` | il mio ipad serve per prendere appunti |
| 639 | `R-S2-0639` | abbiamo ordinato il pollo da kfc |
| 640 | `R-S2-0640` | dopo il logout la sessione scade |
| 641 | `R-S2-0641` | i mondays dopo le ferie sono difficili |
| 642 | `R-S2-0642` | un oral exam è fissato per lunedì |
| 643 | `R-S2-0643` | il popup della privacy blocca il sito |
| 644 | `R-S2-0644` | il prototype ha superato il test |
| 645 | `R-S2-0645` | un reset del modem risolve spesso |
| 646 | `R-S2-0646` | serve un restart dell'applicazione |
| 647 | `R-S2-0647` | il roomba pulisce il soggiorno ogni mattina |
| 648 | `R-S2-0648` | la scaleup cerca nuovi investitori |
| 649 | `R-S2-0649` | allega uno screenshot alla segnalazione |
| 650 | `R-S2-0650` | lo selfie con il team è venuto bene |
| 651 | `R-S2-0651` | lo screen sharing consuma molta banda |
| 652 | `R-S2-0652` | lo shutdown del sistema è automatico |
| 653 | `R-S2-0653` | abbiamo comprato le immagini su shutterstock |
| 654 | `R-S2-0654` | le skills tecniche si imparano sul campo |
| 655 | `R-S2-0655` | il volo più economico era su skyscanner |
| 656 | `R-S2-0656` | lo smartwatch ha la batteria scarica |
| 657 | `R-S2-0657` | il problem solving è la competenza più richiesta |
| 658 | `R-S2-0658` | il caffè di starbucks costa troppo |
| 659 | `R-S2-0659` | la startup ha aperto una sede a milano |
| 660 | `R-S2-0660` | il subway della stazione è chiuso |
| 661 | `R-S2-0661` | lo swipe verso sinistra cancella il messaggio |
| 662 | `R-S2-0662` | le recensioni su tripadvisor erano ottime |
| 663 | `R-S2-0663` | ho comprato il tè twinings al supermercato |
| 664 | `R-S2-0664` | ho fatto unfollow dopo quel post |
| 665 | `R-S2-0665` | il servizio di upload richiede tempo |
| 666 | `R-S2-0666` | il wallpaper nuovo è troppo scuro |
| 667 | `R-S2-0667` | il website non si apre da mobile |
| 668 | `R-S2-0668` | il checkout sul domain nuovo non funziona |
| 669 | `R-S2-0669` | un reset e un restart risolvono il problema |
| 670 | `R-S2-0670` | lo screenshot è nella folder dei bug |
| 671 | `R-S2-0671` | il servizio di upload è lento e lo screen sharing si blocca |
| 672 | `R-S2-0672` | un selfie da starbucks è quasi obbligatorio |
| 673 | `R-S2-0673` | il mio ipad e il charger sono nello zaino |

