import nltk
from nltk.util import ngrams
from collections import defaultdict, Counter
import math

corpus = """Abia trecuseră dar câteva luni după Sf. Gheorghe, şi drumeţii mai umblaţi nu mai ziceau că o să facă
popas la Moara cu noroc, ci că se vor opri la Ghiţă, şi toată lumea ştia cine e Ghiţă şi unde e Ghiţă, iar acolo, în vale,
între pripor şi locurile cele rele, nu mai era Moara cu noroc, ci cârciuma lui Ghiţă.
Iară pentru Ghiţă cârciuma era cu noroc. Patru zile pe săptămână, de marţi seară până sâmbătă, era mereu plină, şi
toţi se opreau la cârciuma lui Ghiţă, şi toţi luau câte ceva, şi toţi plăteau cinstit.
Sâmbătă de cu seară locul se deşerta, şi Ghiţă, ajungând să mai răsufle, se punea cu Ana şi cu bătrâna să numere
banii, şi atunci el privea la Ana, Ana privea la el, amândoi priveau la cei doi copilaşi, căci doi erau acum, iară
bătrâna privea la câteşipatru şi se simţea întreţinută, căci avea un ginere harnic, o fată norocoasă, doi nepoţi
sprinteni, iară sporul era dat de la Dumnezeu, dintr-un câştig făcut cu bine.
Duminică dimineaţă Ghiţă punea calul la teleagă şi bătrâna se ducea la biserică, fiindcă bătrânul, fie iertat, fusese
cojocar şi cântăreţ de strană, şi aşa, mergând la biserică, ea se ducea parcă să-l vadă pe el.
Când bătrâna pleca la biserică, toate trebuiau să fie puse bine la cale, căci altfel ea odată cu capul nu ar fi plecat. Încă
sâmbătă dupăamiazăzi sluga trebuia să rânească grajdul, curtea şi locul de dinaintea cârciumii, în vreme ce bătrâna şi
Ana găteau cârciuma pentru ziua de duminică. Duminică în zori bătrâna primenea copiii, se gătea de sărbătoare, mai
dădea o raită prin împrejur, ca să vadă dacă în adevăr toate sunt bine, apoi se urca în teleagă.
Ana şi Ghiţă îi sărutau mâna, ea mai săruta o dată copilaşii, apoi zicea: "Gând bun să ne dea Dumnezeu!", îşi făcea
cruce şi dădea semn de plecare.
Dar ea pleca totdeauna cu inima grea, căci trebuia să plece singură şi să-i lase pe dânşii singuri la pustietatea aceea
de cârciumă.
Dacă aruncai privirea împrejur, la dreapta şi la stânga, vedeai drumul de ţară şerpuind spre culme, iară la vale, de-a
lungul râuleţului, cât străbate ochiul, până la câmpia nesfârşită, afară de câţiva arini ce stăteau grămadă din jos pe
podul de piatră, nu zăreai decât iarbă şi mărăcini. La deal valea se strâmtează din ce în ce mai mult; dar aici vederile
sunt multe şi deosebite: de-a lungul râuleţului se întind două şiruri de sălcii şi de răchite, care se îndeasă mereu, până
se pierd în crângul din fundul văii; pe culmea dealului de la stânga, despre Ineu, se iveşte pe ici, pe colo marginea
unei păduri de stejar, iară pe dealul de la dreapta stau răzleţe rămăşiţele încă nestârpite ale unei alte păduri, cioate,
rădăcini ieşite din pământ şi, tocmai sus la culme, un trunchi înalt, pe jumătate ars, cu crengile uscate, loc de popas
pentru corbii ce se lasă croncănind de la deal înspre câmpie; fundul văii, în sfârşit, se întunecă, şi din dosul crângului
depărtat iese turnul ţuguiat al bisericii din Fundureni, învelit cu tinichea, dară pierdut oarecum în umbra dealurilor
acoperite cu păduri posomorâte, ce se ridică şi se grămădesc unul peste altul, până la muntele Bihorului, de pe ale
cărui culmi troienite se răsfrâng razele soarelui de dimineaţă.
Rămânând singur cu Ana şi cu copiii, Ghiţă priveşte împrejurul său, se bucură de frumuseţea locului şi inima îi râde
când Ana cea înţeleaptă şi aşezată deodată îşi pierde cumpătul şi se aruncă răsfăţată asupra lui, căci Ana era tânără şi
frumoasă, Ana era fragedă şi subţirică, Ana era sprintenă şi mlădioasă, iară el însuşi, înalt şi spătos, o purta ca pe o
pană subţirică.
Numai câteodată, când în timp de noapte vântul zgâlţâia moara părăsită, locul îi părea lui Ghiţă străin şi pustiicios, şi
atunci el pipăia prin întuneric, ca să vadă dacă Ana, care dormea ca un copil îmbăiat lângă dânsul, nu cumva s-a
descoperit prin somn, şi s-o acopere iar.

Veneau câteodată pe la cârciuma Cât ţin luncile, ele sunt pline de turme de porci, iară unde sunt multe turme, trebuie să fie şi mulţi păstori. Dar şi
porcarii sunt oameni, ba, între mulţi, sunt oameni de tot felul, şi de rând, şi de mâna a doua, ba chiar şi oameni de
frunte.
Veneau câteodată pe la cârciuma Veneau câteodată pe la cârciuma Veneau câteodată pe la cârciuma Veneau câteodată pe la cârciuma Veneau câteodată pe la cârciuma Veneau câteodată pe la cârciuma Veneau câteodată pe la cârciuma O turmă nu poate să fie prea mare, şi aşa, unde sunt mii şi mii de porci, trebuie să fie sute de turme, şi fiecare turmă
are câte un păstor, şi fiecare păstor e ajutat de către doi-trei băieţi, boitarii, adeseori şi mai mulţi, dacă turma e mare.
E dar pe lunci un întreg neam de porcari, oameni care s-au trezit în pădure la turma de grăsuni, ai căror părinţi buni şi
străbuni tot păstori au fost, oameni care au obiceiurile lor şi limba lor păstorească, pe care numai ei o înţeleg. Şi
fiindcă nu-i neguţătorie fără de pagubă, iară păstorii sunt oameni săraci, trebuie să fie cineva care să răspundă de
paguba care se face în turmă: acest cineva este "sămădăul", porcar şi el, dar om cu stare, care poate să plătească
grăsunii pierduţi ori pe cei furaţi. De aceea sămădăul nu e numai om cu stare, ci mai ales om aspru şi neîndurat, care
umblă mereu călare de la turmă la turmă, care ştie toate înfundăturile, cunoaşte pe toţi oamenii buni şi mai ales pe cei
răi, de care tremură toată lunca şi care ştie să afle urechea grăsunului pripăşit chiar şi din oala cu varză.
Şi dacă lumea zice că locurile de lângă Moara cu noroc sunt rele, n-ai fi avut decât să-l întrebi pe vreunul dintre
sămădăi, şi el ţi-ar fi putut spune pentru ce nu sunt bune şi cine le primejduieşte; dar sămădăul e, mai presus de toate,
om tăcut, şi dacă îl întrebi asemenea lucruri, el răspunde: "Nu ştiu, n-am văzut, am atâtea şi atâtea turme în
răspunderea mea şi nu mă pot strica cu oamenii". 
                                                                                El ştie ce ştie
, numai pentru nevoile lui.
Veneau câteodată pe la cârciuma lui Ghiţă şi porcari, nişte oameni îndeobşte înalţi şi bine făcuţi, cu cămaşa neagră şi
cu părul strălucitor de untura cea multă şi căzut în plete lungi şi răsucite asupra grumajilor goi; oameni erau şi ei,
chiar oameni cinstiţi, care mănâncă, beau şi plătesc.
Într-o zi de luni au venit trei inşi în căruţă cu osiile de fier, uşurică şi trasă de doi cai frumoşi, dintre care însă unul
mai mare şi altul mai mic. În căruţă nu era nici scaun, nici fân, ci unul dintre porcarii unsuroşi mâna caii, stând în
picioare, iară ceilalţi doi şedeau pe leutrele vopsite în verde, ca şi când n-ar fi venind decât de aci din apropiere.
"Ăştia nu prea îmi par a oameni buni", îşi zise Ghiţă când îi văzu sărind din căruţă şi privind împrejur, ca unii ce au
mai fost pe aici şi acum nu găsesc nici locul, nici oamenii ca odinioară.
Ei întrebară dacă n-a fost sămădăul pe acolo, puseră sluga să deshame caii, să-i adape şi să le dea ovăz, apoi intrară,
băură fiecare cât trei inşi la un loc şi plecară cu un "noroc bun"."""

def tokenize_text(text):
    return nltk.word_tokenize(text)

def generate_ngrams(tokens, n):
    return list(ngrams(tokens, n))

def count_ngrams(ngrams):
    return Counter(ngrams)

def good_turing_smoothing(ngram_counts):
    frequency_of_frequencies = defaultdict(int)  #n-grams appear c times
    for count in ngram_counts.values():
        frequency_of_frequencies[count] += 1
    
    total_ngrams = sum(ngram_counts.values())
    smoothed_probs = {}
    
    for ngram, count in ngram_counts.items():
        if count == 0:
            continue
        if (count + 1) in frequency_of_frequencies:
            adjusted_count = (count + 1) * (frequency_of_frequencies[count + 1] / frequency_of_frequencies[count])
        else:
            adjusted_count = count

        smoothed_probs[ngram] = adjusted_count / total_ngrams
    
    return smoothed_probs

def kneser_ney_smoothing(ngram_counts, discount=0.75):
    continuation_counts = defaultdict(lambda: defaultdict(int))
    total_ngrams = sum(ngram_counts.values())
    for ngram, count in ngram_counts.items():
        if count > 0:
            continuation_counts[ngram[:-1]][ngram[-1]] += 1

    smoothed_probabilities = {}

    for ngram, count in ngram_counts.items():
        if count > 0:
            discounted_count = max(count - discount, 0)
            continuation_prob = len(continuation_counts[ngram[:-1]]) / total_ngrams if total_ngrams > 0 else 0
            smoothed_probabilities[ngram] = (discounted_count / total_ngrams) + continuation_prob
        else:
            smoothed_probabilities[ngram] = 0.0

    return smoothed_probabilities

def calculate_sentence_probability(sentence, smoothed_probs, n):
    tokens = tokenize_text(sentence)
    ngrams_in_sentence = generate_ngrams(tokens, n)
    log_prob = 0
    
    for ngram in ngrams_in_sentence:
        if ngram in smoothed_probs:
            log_prob += math.log(smoothed_probs[ngram])
        else:
            log_prob += math.log(1e-6)  # aproape 0 daca nu exista
    
    return math.exp(log_prob)

# Main execution
if __name__ == "__main__":
    tokens = tokenize_text(corpus)
    n = 5
    ngrams_list = generate_ngrams(tokens, n)
    ngram_counts = count_ngrams(ngrams_list)

    #smoothed_probs = good_turing_smoothing(ngram_counts)#Good-Turing smoothing
    smoothed_probs = kneser_ney_smoothing(ngram_counts)#Kneser-Ney smoothing

    #TASK 3
    for ngram, prob in smoothed_probs.items():
        print(f"{ngram}: {prob:.4f}")

    #TASK 4
    test_sentence = "Eu ştie ce ştiu"
    prob = calculate_sentence_probability(test_sentence, smoothed_probs, n)

    print(f"Probability of the sentence '{test_sentence}' is: {prob}")