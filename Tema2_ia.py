import time
import copy

import numpy
import pygame
import sys
import random


def elem_identice(lista):
    if (len(set(lista)) == 1):
        return lista[0] if lista[0] != Joc.GOL else False
    return False


class Joc:
    """
    Clasa care defineste jocul. Se va schimba de la un joc la altul.
    """
    JMIN = None
    JMAX = None
    GOL = '#'
    NR_LINII = None
    NR_COLOANE = None
    scor_maxim = 0
    # retine 'x' pentru rosu sau '0' prentru albastru
    jucator_curent = ''
    # retine daca jucatorul alege sa joace impotriva unui AI, impotriva altui jucator sau sa observe doi AI jucand intre ei
    tip_joc = 3

    def __init__(self, matr=None, NR_LINII=None, NR_COLOANE=None, nr_otr=0, poz_otr=(0, 0), tip_joc = 3):
        # creez proprietatea ultima_mutare # (l,c)
        self.ultima_mutare = None
        self.nr_otr = nr_otr
        self.poz_otr = poz_otr

        if matr:
            # e data tabla, deci suntem in timpul jocului
            self.matr = matr
        else:
            # nu e data tabla deci suntem la initializare
            self.matr = [[self.__class__.GOL] * NR_COLOANE for i in range(NR_LINII)]

            if NR_LINII is not None:
                self.__class__.NR_LINII = NR_LINII
            if NR_COLOANE is not None:
                self.__class__.NR_COLOANE = NR_COLOANE

            ######## calculare scor maxim ###########
            sc_randuri = (NR_COLOANE - 3) * NR_LINII
            sc_coloane = (NR_LINII - 3) * NR_COLOANE
            sc_diagonale = (NR_LINII - 3) * (NR_COLOANE - 3) * 2
            self.__class__.scor_maxim = sc_randuri + sc_coloane + sc_diagonale
    # 4. - interfata grafica
    def deseneaza_grid(self, linie_marcaj=None, coloana_marcaj=None):  # tabla de exemplu este ["#","x","#","0",......]

        for ind in range(self.__class__.NR_COLOANE * self.__class__.NR_LINII):
            linie = ind // self.__class__.NR_COLOANE  # // inseamna div
            coloana = ind % self.__class__.NR_COLOANE

            if coloana == coloana_marcaj and linie == linie_marcaj:
                # daca am o patratica selectata, o desenez cu galben
                culoare = (255, 255, 0)
            else:
                # altfel o desenez cu alb
                culoare = (255, 255, 255)
            pygame.draw.rect(self.__class__.display, culoare, self.__class__.celuleGrid[ind])  # alb = (255,255,255)
            if self.matr[linie][coloana] == 'x':
                self.__class__.display.blit(self.__class__.x_img, (coloana * (self.__class__.dim_celula + 1), linie * (self.__class__.dim_celula + 1)))
            elif self.matr[linie][coloana] == '0':
                self.__class__.display.blit(self.__class__.zero_img, (
                coloana * (self.__class__.dim_celula + 1), linie * (self.__class__.dim_celula + 1)))
            elif self.matr[linie][coloana] == 'r':
                self.__class__.display.blit(self.__class__.otr_img, (
                coloana * (self.__class__.dim_celula + 1), linie * (self.__class__.dim_celula + 1)))
        # pygame.display.flip()
        pygame.display.update()

    def deseneaza_grid_sel(self, culoare=(255, 255, 255), linie_marcaj=None, coloana_marcaj=None):  # tabla de exemplu este ["#","x","#","0",......]

        for ind in range(self.__class__.NR_COLOANE * self.__class__.NR_LINII):
            linie = ind // self.__class__.NR_COLOANE  # // inseamna div
            coloana = ind % self.__class__.NR_COLOANE

            if coloana == coloana_marcaj and linie == linie_marcaj:
                # daca am o patratica selectata, o desenez cu galben
                culoare = (252, 3, 3)
            else:
                # altfel o desenez cu alb
                culoare = (255, 255, 255)
            pygame.draw.rect(self.__class__.display, culoare, self.__class__.celuleGrid[ind])  # alb = (255,255,255)
            if self.matr[linie][coloana] == 'x':
                self.__class__.display.blit(self.__class__.x_img, (coloana * (self.__class__.dim_celula + 1), linie * (self.__class__.dim_celula + 1)))
            elif self.matr[linie][coloana] == '0':
                self.__class__.display.blit(self.__class__.zero_img, (
                coloana * (self.__class__.dim_celula + 1), linie * (self.__class__.dim_celula + 1)))
        # pygame.display.flip()
        pygame.display.update()

    @classmethod
    def jucator_opus(cls, jucator):
        return cls.JMAX if jucator == cls.JMIN else cls.JMIN

    @classmethod
    def initializeaza(cls, display, NR_LINII=4, NR_COLOANE=5, dim_celula=100):
        cls.display = display
        cls.dim_celula = dim_celula
        # am adaugat o variabila pentru a putea plasa 'x' uri pe grid, adica patratele otravite
        cls.otr_img = pygame.image.load('otrava.png')
        cls.otr_img = pygame.transform.scale(cls.otr_img, (dim_celula, dim_celula))
        cls.x_img = pygame.image.load('ics.png')
        cls.x_img = pygame.transform.scale(cls.x_img, (dim_celula, dim_celula))
        cls.zero_img = pygame.image.load('zero.png')
        cls.zero_img = pygame.transform.scale(cls.zero_img, (dim_celula, dim_celula))
        cls.celuleGrid = []  # este lista cu patratelele din grid
        for linie in range(NR_LINII):
            for coloana in range(NR_COLOANE):
                patr = pygame.Rect(coloana * (dim_celula + 1), linie * (dim_celula + 1), dim_celula, dim_celula)
                cls.celuleGrid.append(patr)

    def parcurgere(self, directie):
        um = self.ultima_mutare  # (l,c)
        culoare = self.matr[um[0]][um[1]]
        nr_mutari = 0
        while True:
            um = (um[0] + directie[0], um[1] + directie[1])
            if not 0 <= um[0] < self.__class__.NR_LINII or not 0 <= um[1] < self.__class__.NR_COLOANE:
                break
            if not self.matr[um[0]][um[1]] == culoare:
                break
            nr_mutari += 1
        return nr_mutari

    # 7. - stare finala
    # am modificat functia final pentru a testa daca unul din cei 2 jucatori a facut ultima mutare
    def final(self):
        for i in range(self.NR_LINII - 1):
            for j in range(self.NR_COLOANE - 1):
                # fiecare dintre cele 4 variabile reprezinta o bucata dintr-un patrat format din 4 bucati
                # sunt cateva cazuri care trebuiesc testate pentru a vedea daca se mai pot face mutari
                ss = self.matr[i][j]
                sd = self.matr[i][j + 1]
                js = self.matr[i + 1][j]
                jd = self.matr[i + 1][j + 1]
                if (ss == '#' and sd == '#' and js == '#' and jd == '#') or (
                    ss == 'r' and sd == '#' and js == '#' and jd == '#') or (
                    ss == '#' and sd == 'r' and js == '#' and jd == '#') or (
                    ss == '#' and sd == '#' and js == 'r' and jd == '#') or (
                    ss == '#' and sd == '#' and js == '#' and jd == 'r') or (
                    ss == 'r' and sd == 'r' and js == '#' and jd == '#') or (
                    ss == '#' and sd == 'r' and js == 'r' and jd == '#') or (
                    ss == '#' and sd == '#' and js == 'r' and jd == 'r') or (
                    ss == 'r' and sd == '#' and js == '#' and jd == 'r') or (
                    ss == 'r' and sd == '#' and js == 'r' and jd == '#') or (
                    ss == '#' and sd == 'r' and js == '#' and jd == 'r') or (
                    ss == 'r' and sd == 'r' and js == 'r' and jd == '#') or (
                    ss == '#' and sd == 'r' and js == 'r' and jd == 'r') or (
                    ss == 'r' and sd == '#' and js == 'r' and jd == 'r') or (
                    ss == 'r' and sd == 'r' and js == '#' and jd == 'r'):
                    return False
        # cazul particular in care exista fundaturi, acesta nu este inclus de testul anterior
        for i in range(self.NR_LINII):
            for j in range(self.NR_COLOANE):
                # variabilele trebuiesc luate in functie de linii si coloane, este testat daca testul se face la marginea matricii
                # unde index-ul poate iesi din range
                aici = self.matr[i][j]
                if i == 0:
                    ss = 'x'
                else:
                    ss = self.matr[i - 1][j]
                if i == Joc.NR_LINII - 1:
                    jj = 'x'
                else:
                    jj = self.matr[i + 1][j]
                if j == 0:
                    s = 'x'
                else:
                    s = self.matr[i][j - 1]
                if j == Joc.NR_COLOANE - 1:
                    d = 'x'
                else:
                    d = self.matr[i][j + 1]
                l = ['x', '0']
                if aici == '#':
                    if ss in l and jj in l and s in l:
                        return False
                    if ss in l and jj in l and d in l:
                        return False
                    if ss in l and s in l and d in l:
                        return False
                    if d in l and jj in l and s in l:
                        return False
        # intoarce castigatorul, in cazul in care functia a ajuns la acest return si nu s-a oprit la celelalte, care intorc False
        return Joc.jucator_curent

    def mutari(self, jucator):
        l_mutari = []
        for j in range(self.__class__.NR_COLOANE):
            last_poz = None
            if self.matr[0][j] != self.__class__.GOL:
                continue
            for i in range(self.__class__.NR_LINII):
                if self.matr[i][j] != self.__class__.GOL:
                    last_poz = (i - 1, j)
                    break
            if last_poz is None:
                last_poz = (self.__class__.NR_LINII - 1, j)
            matr_tabla_noua = copy.deepcopy(self.matr)
            matr_tabla_noua[last_poz[0]][last_poz[1]] = jucator
            jn = Joc(matr_tabla_noua)
            jn.ultima_mutare = (last_poz[0], last_poz[1])
            l_mutari.append(jn)
        return l_mutari

    # linie deschisa inseamna linie pe care jucatorul mai poate forma o configuratie castigatoare
    # practic e o linie fara simboluri ale jucatorului opus
    def linie_deschisa(self, lista, jucator):
        jo = self.jucator_opus(jucator)
        # verific daca pe linia data nu am simbolul jucatorului opus
        if not jo in lista:
            # return 1
            return lista.count(jucator)
        return 0

    def linii_deschise(self, jucator):

        linii = 0
        for i in range(self.__class__.NR_LINII):
            for j in range(self.__class__.NR_COLOANE - 3):
                linii += self.linie_deschisa(self.matr[i][j:j + 4], jucator)

        for j in range(self.__class__.NR_COLOANE):
            for i in range(self.__class__.NR_LINII - 3):
                linii += self.linie_deschisa([self.matr[k][j] for k in range(i, i + 4)], jucator)

        # \
        for i in range(self.__class__.NR_LINII - 3):
            for j in range(self.__class__.NR_COLOANE - 3):
                linii += self.linie_deschisa([self.matr[i + k][j + k] for k in range(0, 4)], jucator)

        # /
        for i in range(self.__class__.NR_LINII - 3):
            for j in range(3, self.__class__.NR_COLOANE):
                linii += self.linie_deschisa([self.matr[i + k][j - k] for k in range(0, 4)], jucator)

        return linii

        """return (self.linie_deschisa(self.matr[0:3],jucator) 
            + self.linie_deschisa(self.matr[3:6], jucator) 
            + self.linie_deschisa(self.matr[6:9], jucator)
            + self.linie_deschisa(self.matr[0:9:3], jucator)
            + self.linie_deschisa(self.matr[1:9:3], jucator)
            + self.linie_deschisa(self.matr[2:9:3], jucator)
            + self.linie_deschisa(self.matr[0:9:4], jucator) #prima diagonala
            + self.linie_deschisa(self.matr[2:8:2], jucator)) # a doua diagonala
        """

    def estimeaza_scor(self, adancime):
        t_final = self.final()
        # if (adancime==0):
        if t_final == self.__class__.JMAX:
            return (self.__class__.scor_maxim + adancime)
        elif t_final == self.__class__.JMIN:
            return (-self.__class__.scor_maxim - adancime)
        elif t_final == 'remiza':
            return 0
        else:
            return (self.linii_deschise(self.__class__.JMAX) - self.linii_deschise(self.__class__.JMIN))

    def sirAfisare(self):
        sir = "  |"
        sir += " ".join([str(i) for i in range(self.NR_COLOANE)]) + "\n"
        sir += "-" * (self.NR_COLOANE + 1) * 2 + "\n"
        sir += "\n".join([str(i) + " |" + " ".join([str(x) for x in self.matr[i]]) for i in range(len(self.matr))])
        return sir

    def __str__(self):
        return self.sirAfisare()

    def __repr__(self):
        return self.sirAfisare()


class Stare:
    """
    Clasa folosita de algoritmii minimax si alpha-beta
    Are ca proprietate tabla de joc
    Functioneaza cu conditia ca in cadrul clasei Joc sa fie definiti JMIN si JMAX (cei doi jucatori posibili)
    De asemenea cere ca in clasa Joc sa fie definita si o metoda numita mutari() care ofera lista cu configuratiile posibile in urma mutarii unui jucator
    """

    def __init__(self, tabla_joc, j_curent, adancime, parinte=None, scor=None):
        self.tabla_joc = tabla_joc
        self.j_curent = j_curent

        # adancimea in arborele de stari
        self.adancime = adancime

        # scorul starii (daca e finala) sau al celei mai bune stari-fiice (pentru jucatorul curent)
        self.scor = scor

        # lista de mutari posibile din starea curenta
        self.mutari_posibile = []

        # cea mai buna mutare din lista de mutari posibile pentru jucatorul curent
        self.stare_aleasa = None

    def mutari(self):
        l_mutari = self.tabla_joc.mutari(self.j_curent)
        juc_opus = Joc.jucator_opus(self.j_curent)
        l_stari_mutari = [Stare(mutare, juc_opus, self.adancime - 1, parinte=self) for mutare in l_mutari]

        return l_stari_mutari

    def __str__(self):
        sir = str(self.tabla_joc) + "(Juc curent:" + self.j_curent + ")\n"
        return sir

    def __repr__(self):
        sir = str(self.tabla_joc) + "(Juc curent:" + self.j_curent + ")\n"
        return sir


""" Algoritmul MinMax """


def min_max(stare):
    if stare.adancime == 0 or stare.tabla_joc.final():
        stare.scor = stare.tabla_joc.estimeaza_scor(stare.adancime)
        return stare

    # calculez toate mutarile posibile din starea curenta
    stare.mutari_posibile = stare.mutari()

    # aplic algoritmul minimax pe toate mutarile posibile (calculand astfel subarborii lor)
    mutari_scor = [min_max(mutare) for mutare in stare.mutari_posibile]

    if stare.j_curent == Joc.JMAX:
        # daca jucatorul e JMAX aleg starea-fiica cu scorul maxim
        stare.stare_aleasa = max(mutari_scor, key=lambda x: x.scor)
    else:
        # daca jucatorul e JMIN aleg starea-fiica cu scorul minim
        stare.stare_aleasa = min(mutari_scor, key=lambda x: x.scor)
    stare.scor = stare.stare_aleasa.scor
    return stare


def alpha_beta(alpha, beta, stare):
    if stare.adancime == 0 or stare.tabla_joc.final():
        stare.scor = stare.tabla_joc.estimeaza_scor(stare.adancime)
        return stare

    if alpha > beta:
        return stare  # este intr-un interval invalid deci nu o mai procesez

    stare.mutari_posibile = stare.mutari()

    if stare.j_curent == Joc.JMAX:
        scor_curent = float('-inf')

        for mutare in stare.mutari_posibile:
            # calculeaza scorul
            stare_noua = alpha_beta(alpha, beta, mutare)

            if (scor_curent < stare_noua.scor):
                stare.stare_aleasa = stare_noua
                scor_curent = stare_noua.scor
            if (alpha < stare_noua.scor):
                alpha = stare_noua.scor
                if alpha >= beta:
                    break

    elif stare.j_curent == Joc.JMIN:
        scor_curent = float('inf')

        for mutare in stare.mutari_posibile:

            stare_noua = alpha_beta(alpha, beta, mutare)

            if (scor_curent > stare_noua.scor):
                stare.stare_aleasa = stare_noua
                scor_curent = stare_noua.scor

            if (beta > stare_noua.scor):
                beta = stare_noua.scor
                if alpha >= beta:
                    break
    stare.scor = stare.stare_aleasa.scor

    return stare

# 7. - stare finala
# coloreaza toate celulele in culoarea castigatorului
def afis_castigator(final, stare_curenta):
    for _ in range(Joc.NR_LINII):
        for __ in range(Joc.NR_COLOANE):
            stare_curenta.tabla_joc.matr[_][__] = final
            stare_curenta.tabla_joc.deseneaza_grid(
                linie_marcaj=_,
                coloana_marcaj=__)

# in cazul in care functia final() intoarce o valoare diferita de False, afis_daca_final afiseaza
# in consola castigatorul si apeleaza functia afis_castigator
def afis_daca_final(stare_curenta):
    final = stare_curenta.tabla_joc.final()
    if (final):
        if (final == "remiza"):
            print("Remiza!")
        else:
            if final == 'x':
                print("A castigat rosu")
            else:
                print("A castigat albastru")
            afis_castigator(final, stare_curenta)

        return True

    return False


class Buton:
    def __init__(self, display=None, left=0, top=0, w=0, h=0, culoareFundal=(53, 80, 115),
                 culoareFundalSel=(89, 134, 194), text="", font="arial", fontDimensiune=16, culoareText=(255, 255, 255),
                 valoare=""):
        self.display = display
        self.culoareFundal = culoareFundal
        self.culoareFundalSel = culoareFundalSel
        self.text = text
        self.font = font
        self.w = w
        self.h = h
        self.selectat = False
        self.fontDimensiune = fontDimensiune
        self.culoareText = culoareText
        # creez obiectul font
        fontObj = pygame.font.SysFont(self.font, self.fontDimensiune)
        self.textRandat = fontObj.render(self.text, True, self.culoareText)
        self.dreptunghi = pygame.Rect(left, top, w, h)
        # aici centram textul
        self.dreptunghiText = self.textRandat.get_rect(center=self.dreptunghi.center)
        self.valoare = valoare

    def selecteaza(self, sel):
        self.selectat = sel
        self.deseneaza()

    def selecteazaDupacoord(self, coord):
        if self.dreptunghi.collidepoint(coord):
            self.selecteaza(True)
            return True
        return False

    def updateDreptunghi(self):
        self.dreptunghi.left = self.left
        self.dreptunghi.top = self.top
        self.dreptunghiText = self.textRandat.get_rect(center=self.dreptunghi.center)

    def deseneaza(self):
        culoareF = self.culoareFundalSel if self.selectat else self.culoareFundal
        pygame.draw.rect(self.display, culoareF, self.dreptunghi)
        self.display.blit(self.textRandat, self.dreptunghiText)


class GrupButoane:
    def __init__(self, listaButoane=[], indiceSelectat=0, spatiuButoane=10, left=0, top=0):
        self.listaButoane = listaButoane
        self.indiceSelectat = indiceSelectat
        self.listaButoane[self.indiceSelectat].selectat = True
        self.top = top
        self.left = left
        leftCurent = self.left
        for b in self.listaButoane:
            b.top = self.top
            b.left = leftCurent
            b.updateDreptunghi()
            leftCurent += (spatiuButoane + b.w)

    def selecteazaDupacoord(self, coord):
        for ib, b in enumerate(self.listaButoane):
            if b.selecteazaDupacoord(coord):
                self.listaButoane[self.indiceSelectat].selecteaza(False)
                self.indiceSelectat = ib
                return True
        return False

    def deseneaza(self):
        # atentie, nu face wrap
        for b in self.listaButoane:
            b.deseneaza()

    def getValoare(self):
        return self.listaButoane[self.indiceSelectat].valoare


############# ecran initial ########################
 # 1. - implementare meniu
def deseneaza_alegeri(display, tabla_curenta):

    btn_alg = GrupButoane(
        top=10,
        left=30,
        listaButoane=[
            Buton(display=display, w=80, h=30, text="minimax", valoare="minimax"),
            Buton(display=display, w=80, h=30, text="alphabeta", valoare="alphabeta")
        ],
        indiceSelectat=1)
    btn_juc = GrupButoane(
        top=50,
        left=30,
        listaButoane=[
            Buton(display=display, w=50, h=30, text="rosu", valoare="x"),
            Buton(display=display, w=50, h=30, text="albastru", valoare="0")
        ],
        indiceSelectat=0)
    # am adaugat un buton pentru dificultate
    # 2. - dificultate
    btn_dif = GrupButoane(
        top=90,
        left=30,
        listaButoane=[
            Buton(display=display, w=50, h=30, text="usor", valoare="1"),
            Buton(display=display, w=50, h=30, text="mediu", valoare="2"),
            Buton(display=display, w=50, h=30, text="greu", valoare="4")
        ],
        indiceSelectat=0)
    # am adaugat un buton pentru alesul numarului de bucati otravite
    btn_otr = GrupButoane(
        top=130,
        left=30,
        listaButoane=[
            Buton(display=display, w=65, h=30, text="3 otravuri", valoare="3"),
            Buton(display=display, w=65, h=30, text="4 otravuri", valoare="4"),
            Buton(display=display, w=65, h=30, text="5 otravuri", valoare="5")
        ],
        indiceSelectat=0)
    # 13. - optiuni jucatori
    # am adaugat un buton pentru alegerea jucatorilor
    btn_tip_juc = GrupButoane(
        top=170,
        left=30,
        listaButoane=[
            Buton(display=display, w=60, h=30, text="AI vs AI", valoare="1"),
            Buton(display=display, w=60, h=30, text="P vs AI", valoare="2"),
            Buton(display=display, w=60, h=30, text="P vs P", valoare="3")
        ],
        indiceSelectat=0)
    ok = Buton(display=display, top=210, left=30, w=40, h=30, text="ok", culoareFundal=(155, 0, 55))
    btn_alg.deseneaza()
    btn_juc.deseneaza()
    btn_dif.deseneaza()
    btn_otr.deseneaza()
    btn_tip_juc.deseneaza()
    ok.deseneaza()
    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif ev.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                if not btn_alg.selecteazaDupacoord(pos):
                    if not btn_juc.selecteazaDupacoord(pos):
                        if not btn_dif.selecteazaDupacoord(pos):
                            if not btn_otr.selecteazaDupacoord(pos):
                                if not btn_tip_juc.selecteazaDupacoord(pos):
                                    if ok.selecteazaDupacoord(pos):
                                        display.fill((0, 0, 0))  # stergere ecran
                                        tabla_curenta.deseneaza_grid()
                                        return btn_juc.getValoare(), btn_alg.getValoare(), btn_dif.getValoare(), btn_otr.getValoare(), btn_tip_juc.getValoare()
        pygame.display.update()

# alege niste valori aleatorii pentru pozitiile bucatilor otravite
def plaseaza_otr(nr_otr, matr, lin, col):
    nr_otr = int(nr_otr)
    while nr_otr != 0:
        x = random.randrange(0, lin - 1)
        y = random.randrange(0, col - 1)
        # testare pentru cazul in care alege aceeasi bucata de mai multe ori
        if matr[x][y] == '#':
            nr_otr = nr_otr - 1
            matr[x][y] = 'r'
    # (x, y) reprezinta pozitiile ultimei bucati otravite alese
    return matr, (x, y)


def main():
    # 3. - generarea starii initiale
    # setari interf grafica

    pygame.init()
    pygame.mixer.init()

    # numele ferestrei in care se deschide jocul
    pygame.display.set_caption("Pascu Stefan-Liviu: Jocul Hap! (Chomp!)")

    # dimensiunea ferestrei in celule
    nl = 9
    nc = 10
    # dimensiunea unei celule
    w = 50

    ecran = pygame.display.set_mode(size=(nc * (w + 1) - 1, nl * (w + 1) - 1))  # N *w+ N-1= N*(w+1)-1
    Joc.initializeaza(ecran, NR_LINII=nl, NR_COLOANE=nc, dim_celula=w)

    # initializare tabla
    tabla_curenta = Joc(NR_LINII=nl, NR_COLOANE=nc)
    # nr_otr = numarul de patratele otravite alese din input
    # tip_joc = jucatorii alesi din input

    # 1. - implementare meniu
    # 2. - dificultate
    # 13. - optiuni jucatori
    Joc.JMIN, tip_algoritm, dificultate, nr_otr, tip_joc = deseneaza_alegeri(ecran, tabla_curenta)
    Joc.nr_otr = nr_otr

    tabla_curenta.matr, poz_otr = plaseaza_otr(nr_otr, tabla_curenta.matr, nl, nc)
    tabla_curenta.poz_otr = poz_otr
    # seteaza adancimea arborelui in functie dificultatea aleasa, jocul devine mai greu cand arborele este mai adanc
    ADANCIME_MAX = int(dificultate)

    # decide fiecare jucator si culoarea acestuia
    if Joc.JMIN == 'x':
        Joc.JMAX = '0'
    else:
        Joc.JMAX = 'x'

    Joc.tip_joc = tip_joc

    print("Tabla initiala")
    print(str(tabla_curenta))

    # creare stare initiala
    stare_curenta = Stare(tabla_curenta, 'x', ADANCIME_MAX)

    # 4. - interfata grafica
    tabla_curenta.deseneaza_grid()

    selectat = 0
    test_timp = 0
    while True:
        # inceperea timpului de gandire
        # 9.a) -afisarea timpului de gandire
        if test_timp == 0:
            t_inainteom = int(round(time.time() * 1000))
            test_timp = 1


        # randul jucatorului
        if (stare_curenta.j_curent == Joc.JMIN):
            for event in pygame.event.get():

                if event.type == pygame.QUIT:
                    # iesim din program
                    pygame.quit()
                    sys.exit()
                #colorarea bucatii peste care se misca mouse-ul
                if event.type == pygame.MOUSEMOTION:

                    pos = pygame.mouse.get_pos()  # coordonatele cursorului
                    for np in range(len(Joc.celuleGrid)):
                        if Joc.celuleGrid[np].collidepoint(pos):
                            stare_curenta.tabla_joc.deseneaza_grid(linie_marcaj=np // Joc.NR_COLOANE, coloana_marcaj = np % Joc.NR_COLOANE)
                            break
                # selectarea celor doua pozitii care decid forma si marimea patrulaterului format din una sau mai multe bucati
                elif event.type == pygame.MOUSEBUTTONDOWN and selectat == 0:
                    # o variabila care face posibila apasarea a doua celule diferite
                    selectat = 1
                    # pozitia primului click
                    pos_start = pygame.mouse.get_pos()  # coordonatele cursorului la momentul clickului

                elif event.type == pygame.MOUSEBUTTONDOWN and selectat == 1:
                    selectat = 0
                    # pozitia celui de-al doilea click
                    pos_end = pygame.mouse.get_pos() # coordonatele cursorului la momentul eliberarii clickului

                    # cazurile in functie de modul in care jucatorul a ales zona
                    # acestea difera in functie de primul si al doilea click
                    # voi adauga comentarii pentru un singur 'if', celelalte sunt analog acestuia
                    if pos_start[0] <= pos_end[0] and pos_start[1] <= pos_end[1]:
                        # are valoarea 0 in cazul in care nu este nicio problema la selectarea patrulaterului de bucatele
                        # in caz contrar va avea valoarea 1
                        test1 = 0
                        # teasteaza daca se afla cel putin o bucata otravita sau deja aleasa de unul dintre jucatori
                        # 6. - functie de testare a validitatii unei mutari
                        for np1 in range(len(Joc.celuleGrid)):
                            if Joc.celuleGrid[np1].collidepoint((pos_start[0], pos_start[1])) and test1 == 0:
                                for np2 in range(len(Joc.celuleGrid)):
                                    if Joc.celuleGrid[np2].collidepoint(pos_end) and test1 == 0:
                                        for g in range(np1, np2 + 1):
                                            # np1 este celula pe care incepe selectarea, np2 este zona in care ia sfarsit
                                            # acest 'if' are scopul de a delimita coloanele patrulaterului ales si de a vedea daca exista
                                            # bucati ce nu ar trebui selecatate
                                            if (g // Joc.NR_COLOANE <= np2 // Joc.NR_COLOANE and g % Joc.NR_COLOANE <= np2 % Joc.NR_COLOANE) and (g // Joc.NR_COLOANE >= np1 // Joc.NR_COLOANE and g % Joc.NR_COLOANE >= np1 % Joc.NR_COLOANE) and (
                                                    (stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][g % Joc.NR_COLOANE] == 'r') or (stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][g % Joc.NR_COLOANE] == '0') or (stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][g % Joc.NR_COLOANE] == 'x')):
                                                test1 = 1
                                                print("Nu puteti manca zona aleasa deoarece contine cel putin o bucata otravita sau aleasa deja!")
                                                break
                        # testeaza daca zona aleasa ar imparti grid-ul in doua bucati neconectate sau daca jucatorul
                        # alege o zona neconectata de margini sau de alta zona deja aleasa
                        # 6. - functie de testare a validitatii unei mutari
                        if test1 == 0:
                            # variabila are rolul de a numara de cate ori se modifica elementele dintr-un vector format
                            # din marginea patrulaterului ales
                            nri = 0
                            #  in aceste doua 'for-uri' este compusa marginea patrulaterului selectat
                            for np1 in range(len(Joc.celuleGrid)):
                                if Joc.celuleGrid[np1].collidepoint((pos_start[0], pos_start[1])) and test1 == 0:
                                    np = np1
                                    su = 0
                                    st = 0
                                    if np1 >= Joc.NR_COLOANE:
                                        su = 1
                                        np = np - Joc.NR_COLOANE
                                    if np1 % Joc.NR_COLOANE != 0:
                                        st = 1
                                        np = np - 1
                                    np1 = np
                                    i1 = 0
                                    i2 = Joc.NR_LINII + Joc.NR_COLOANE + 5
                                    for np2 in range(len(Joc.celuleGrid)):
                                        if Joc.celuleGrid[np2].collidepoint(pos_end) and test1 == 0:
                                            np = np2
                                            jo = 0
                                            dr = 0
                                            if np2 <= (Joc.NR_COLOANE * Joc.NR_LINII) - Joc.NR_COLOANE - 1:
                                                jo = 1
                                                np = np + Joc.NR_COLOANE
                                            if (np2 + 1) % Joc.NR_COLOANE != 0:
                                                dr = 1
                                                np = np + 1
                                            np2 = np
                                            i3 = Joc.NR_LINII + Joc.NR_COLOANE * 2 + 8
                                            i4 = Joc.NR_LINII + Joc.NR_COLOANE * 2 + 9
                                            # lista in care sunt introduse valorile din marginea patrulaterului
                                            listm = [-1 for _ in range((Joc.NR_COLOANE + Joc.NR_LINII + 20) * 2 + 1)]
                                            for g in range(np1, np2 + 1):
                                                # fiecare 'if' are rolul de a se 'ocupa' de o margine: sus, jos, stanga, dreapta
                                                if (
                                                        g // Joc.NR_COLOANE == np2 // Joc.NR_COLOANE and g % Joc.NR_COLOANE <= np2 % Joc.NR_COLOANE) and (
                                                        g // Joc.NR_COLOANE >= np1 // Joc.NR_COLOANE and g % Joc.NR_COLOANE >= np1 % Joc.NR_COLOANE):
                                                    if jo == 1:
                                                        stare_curenta.tabla_joc.deseneaza_grid(
                                                                linie_marcaj=g // Joc.NR_COLOANE,
                                                                coloana_marcaj=g % Joc.NR_COLOANE)
                                                        if stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][g % Joc.NR_COLOANE] == 'x' or stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][g % Joc.NR_COLOANE] == '0':
                                                            listm[i1] = 'x'  # 'x'
                                                        else:
                                                            listm[i1] = '#'
                                                        i1 += 1
                                                    else:
                                                        listm[i1] = 'x'
                                                        i1 += 1

                                                if (
                                                        g // Joc.NR_COLOANE <= np2 // Joc.NR_COLOANE and g % Joc.NR_COLOANE == np2 % Joc.NR_COLOANE) and (
                                                        g // Joc.NR_COLOANE >= np1 // Joc.NR_COLOANE and g % Joc.NR_COLOANE >= np1 % Joc.NR_COLOANE):
                                                    if dr == 1:

                                                        stare_curenta.tabla_joc.deseneaza_grid(
                                                                linie_marcaj=g // Joc.NR_COLOANE,
                                                                coloana_marcaj=g % Joc.NR_COLOANE)
                                                        if stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][g % Joc.NR_COLOANE] == 'x' or stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][g % Joc.NR_COLOANE] == '0':
                                                            listm[i2] = 'x'
                                                        else:
                                                            listm[i2] = '#'
                                                        i2 -= 1
                                                    else:
                                                        listm[i2] = 'x'
                                                        i2 -= 1

                                                if (
                                                        g // Joc.NR_COLOANE <= np2 // Joc.NR_COLOANE and g % Joc.NR_COLOANE <= np2 % Joc.NR_COLOANE) and (
                                                        g // Joc.NR_COLOANE == np1 // Joc.NR_COLOANE and g % Joc.NR_COLOANE >= np1 % Joc.NR_COLOANE):
                                                    if su == 1:
                                                        stare_curenta.tabla_joc.deseneaza_grid(
                                                                linie_marcaj=g // Joc.NR_COLOANE,
                                                                coloana_marcaj=g % Joc.NR_COLOANE)
                                                        if stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][g % Joc.NR_COLOANE] == 'x' or stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][g % Joc.NR_COLOANE] == '0':
                                                            listm[i3] = 'x'
                                                        else:
                                                            listm[i3] = '#'
                                                        i3 -= 1
                                                    else:
                                                        listm[i3] = 'x'
                                                        i3 -= 1

                                                if (
                                                        g // Joc.NR_COLOANE <= np2 // Joc.NR_COLOANE and g % Joc.NR_COLOANE <= np2 % Joc.NR_COLOANE) and (
                                                        g // Joc.NR_COLOANE >= np1 // Joc.NR_COLOANE and g % Joc.NR_COLOANE == np1 % Joc.NR_COLOANE):
                                                    if st == 1:
                                                        stare_curenta.tabla_joc.deseneaza_grid(
                                                                linie_marcaj=g // Joc.NR_COLOANE,
                                                                coloana_marcaj=g % Joc.NR_COLOANE)
                                                        if stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][g % Joc.NR_COLOANE] == 'x' or stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][g % Joc.NR_COLOANE] == '0':
                                                            listm[i4] = 'x'
                                                        else:
                                                            listm[i4] = '#'
                                                        i4 += 1
                                                    else:
                                                        listm[i4] = 'x'
                                                        i4 += 1

                            listm = list(filter(lambda a: a != -1, listm))
                            listm.append(listm[0])
                            # parcurgerea listei
                            for _ in range(len(listm) - 1):
                                # marirea lui nri in functie de modificarile de variabile din lista
                                if (listm[_] == 'x' and listm[_ + 1] == '#') or (listm[_] == '#' and listm[_ + 1] == 'x'):
                                    nri += 1
                            # daca nri este mai mare decat doi, inseamna ca patrulaterul ales imparte grid-ul in
                            # doua sau mai multe parti, daca nri este mai mic sau egal cu zero, patrulaterul se
                            # afla intr-o zona neconectata cu alte zone deja alese sau cu marginea
                            if nri > 2 or nri <= 0:
                                test1 = 1
                        # odata ajuns aici, daca test1 este zero, algoritmul deseneaza in matrice si in grid zona aleasa
                        # 7. - stare finala
                        if test1 == 0:
                            for np1 in range(len(Joc.celuleGrid)):
                                if Joc.celuleGrid[np1].collidepoint((pos_start[0], pos_start[1])):
                                    for np2 in range(len(Joc.celuleGrid)):
                                        if Joc.celuleGrid[np2].collidepoint(pos_end):
                                            for g in range(np1, np2 + 1):
                                                if (g // Joc.NR_COLOANE <= np2 // Joc.NR_COLOANE and g % Joc.NR_COLOANE <= np2 % Joc.NR_COLOANE) and (g // Joc.NR_COLOANE >= np1 // Joc.NR_COLOANE and g % Joc.NR_COLOANE >= np1 % Joc.NR_COLOANE):
                                                    stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][g % Joc.NR_COLOANE] = Joc.JMIN
                                                    stare_curenta.tabla_joc.ultima_mutare = (g - np1, g % Joc.NR_COLOANE)
                                                    # print(str(stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][g % Joc.NR_COLOANE]))
                                                    stare_curenta.tabla_joc.deseneaza_grid(linie_marcaj=g // Joc.NR_COLOANE,
                                                                                   coloana_marcaj=g % Joc.NR_COLOANE)
                                            break
                            print("\nTabla dupa mutarea jucatorului")
                            print(str(stare_curenta))

                    if pos_start[0] < pos_end[0] and pos_start[1] > pos_end[1]:
                        test1 = 0
                        for np1 in range(len(Joc.celuleGrid)):
                            if Joc.celuleGrid[np1].collidepoint((pos_start[0], pos_end[1])) and test1 == 0:
                                for np2 in range(len(Joc.celuleGrid)):
                                    if Joc.celuleGrid[np2].collidepoint((pos_end[0], pos_start[1])) and test1 == 0:
                                        for g in range(np1, np2 + 1):
                                            if (g // Joc.NR_COLOANE <= np2 // Joc.NR_COLOANE and g % Joc.NR_COLOANE <= np2 % Joc.NR_COLOANE) and (
                                                    g // Joc.NR_COLOANE >= np1 // Joc.NR_COLOANE and g % Joc.NR_COLOANE >= np1 % Joc.NR_COLOANE) and (
                                                    (stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][
                                                        g % Joc.NR_COLOANE] == 'r')  or (stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][g % Joc.NR_COLOANE] == '0') or (stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][g % Joc.NR_COLOANE] == 'x')):
                                                test1 = 1
                                                print("Nu puteti manca zona aleasa deoarece contine cel putin o bucata otravita sau aleasa deja!")
                                                break

                        if test1 == 0:
                            nri = 0
                            for np1 in range(len(Joc.celuleGrid)):
                                if Joc.celuleGrid[np1].collidepoint((pos_start[0], pos_end[1])) and test1 == 0:
                                    np = np1
                                    su = 0
                                    st = 0
                                    if np1 >= Joc.NR_COLOANE:
                                        su = 1
                                        np = np - Joc.NR_COLOANE
                                    if np1 % Joc.NR_COLOANE != 0:
                                        st = 1
                                        np = np - 1
                                    np1 = np
                                    i1 = 0
                                    i2 = Joc.NR_LINII + Joc.NR_COLOANE + 5
                                    # print(str(i1) + " " + str(i2))
                                    for np2 in range(len(Joc.celuleGrid)):
                                        if Joc.celuleGrid[np2].collidepoint(pos_end[0], pos_start[1]) and test1 == 0:
                                            np = np2
                                            jo = 0
                                            dr = 0
                                            if np2 <= (Joc.NR_COLOANE * Joc.NR_LINII) - Joc.NR_COLOANE - 1:
                                                jo = 1
                                                np = np + Joc.NR_COLOANE
                                            if (np2 + 1) % Joc.NR_COLOANE != 0:
                                                dr = 1
                                                np = np + 1
                                            np2 = np
                                            i3 = Joc.NR_LINII + Joc.NR_COLOANE * 2 + 8
                                            i4 = Joc.NR_LINII + Joc.NR_COLOANE * 2 + 9
                                            listm = [-1 for _ in range((Joc.NR_COLOANE + Joc.NR_LINII + 20) * 2 + 1)]
                                            for g in range(np1, np2 + 1):
                                                if (
                                                        g // Joc.NR_COLOANE == np2 // Joc.NR_COLOANE and g % Joc.NR_COLOANE <= np2 % Joc.NR_COLOANE) and (
                                                        g // Joc.NR_COLOANE >= np1 // Joc.NR_COLOANE and g % Joc.NR_COLOANE >= np1 % Joc.NR_COLOANE):
                                                    if jo == 1:
                                                        stare_curenta.tabla_joc.deseneaza_grid(
                                                                linie_marcaj=g // Joc.NR_COLOANE,
                                                                coloana_marcaj=g % Joc.NR_COLOANE)
                                                        if stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][g % Joc.NR_COLOANE] == 'x' or stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][g % Joc.NR_COLOANE] == '0':
                                                            listm[i1] = 'x'  # 'x'
                                                        else:
                                                            listm[i1] = '#'
                                                        i1 += 1
                                                    else:
                                                        listm[i1] = 'x'
                                                        i1 += 1

                                                if (
                                                        g // Joc.NR_COLOANE <= np2 // Joc.NR_COLOANE and g % Joc.NR_COLOANE == np2 % Joc.NR_COLOANE) and (
                                                        g // Joc.NR_COLOANE >= np1 // Joc.NR_COLOANE and g % Joc.NR_COLOANE >= np1 % Joc.NR_COLOANE):
                                                    if dr == 1:

                                                        stare_curenta.tabla_joc.deseneaza_grid(
                                                                linie_marcaj=g // Joc.NR_COLOANE,
                                                                coloana_marcaj=g % Joc.NR_COLOANE)
                                                        if stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][g % Joc.NR_COLOANE] == 'x' or stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][g % Joc.NR_COLOANE] == '0':
                                                            listm[i2] = 'x'
                                                        else:
                                                            listm[i2] = '#'
                                                        i2 -= 1
                                                    else:
                                                        listm[i2] = 'x'
                                                        i2 -= 1

                                                if (
                                                        g // Joc.NR_COLOANE <= np2 // Joc.NR_COLOANE and g % Joc.NR_COLOANE <= np2 % Joc.NR_COLOANE) and (
                                                        g // Joc.NR_COLOANE == np1 // Joc.NR_COLOANE and g % Joc.NR_COLOANE >= np1 % Joc.NR_COLOANE):
                                                    if su == 1:
                                                        stare_curenta.tabla_joc.deseneaza_grid(
                                                                linie_marcaj=g // Joc.NR_COLOANE,
                                                                coloana_marcaj=g % Joc.NR_COLOANE)
                                                        if stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][g % Joc.NR_COLOANE] == 'x' or stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][g % Joc.NR_COLOANE] == '0':
                                                            listm[i3] = 'x'
                                                        else:
                                                            listm[i3] = '#'
                                                        i3 -= 1
                                                    else:
                                                        listm[i3] = 'x'
                                                        i3 -= 1

                                                if (
                                                        g // Joc.NR_COLOANE <= np2 // Joc.NR_COLOANE and g % Joc.NR_COLOANE <= np2 % Joc.NR_COLOANE) and (
                                                        g // Joc.NR_COLOANE >= np1 // Joc.NR_COLOANE and g % Joc.NR_COLOANE == np1 % Joc.NR_COLOANE):
                                                    if st == 1:
                                                        stare_curenta.tabla_joc.deseneaza_grid(
                                                                linie_marcaj=g // Joc.NR_COLOANE,
                                                                coloana_marcaj=g % Joc.NR_COLOANE)
                                                        if stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][g % Joc.NR_COLOANE] == 'x' or stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][g % Joc.NR_COLOANE] == '0':
                                                            listm[i4] = 'x'
                                                        else:
                                                            listm[i4] = '#'
                                                        i4 += 1
                                                    else:
                                                        listm[i4] = 'x'
                                                        i4 += 1

                            listm = list(filter(lambda a: a != -1, listm))
                            listm.append(listm[0])
                            # print(listm)
                            for _ in range(len(listm) - 1):
                                # print(_)
                                if (listm[_] == 'x' and listm[_ + 1] == '#') or (listm[_] == '#' and listm[_ + 1] == 'x'):
                                    nri += 1
                            if nri > 2 or nri <= 0:
                                test1 = 1

                        if test1 == 0:
                            for np1 in range(len(Joc.celuleGrid)):
                                if Joc.celuleGrid[np1].collidepoint((pos_start[0], pos_end[1])):
                                    for np2 in range(len(Joc.celuleGrid)):
                                        if Joc.celuleGrid[np2].collidepoint((pos_end[0], pos_start[1])):
                                            for g in range(np1, np2 + 1):
                                                if (g // Joc.NR_COLOANE <= np2 // Joc.NR_COLOANE and g % Joc.NR_COLOANE <= np2 % Joc.NR_COLOANE) and (g // Joc.NR_COLOANE >= np1 // Joc.NR_COLOANE and g % Joc.NR_COLOANE >= np1 % Joc.NR_COLOANE):
                                                    stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][g % Joc.NR_COLOANE] = Joc.JMIN
                                                    stare_curenta.tabla_joc.ultima_mutare = (g - np1, g % Joc.NR_COLOANE)
                                                    # print(str(stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][g % Joc.NR_COLOANE]))
                                                    stare_curenta.tabla_joc.deseneaza_grid(linie_marcaj=g // Joc.NR_COLOANE,
                                                                                   coloana_marcaj=g % Joc.NR_COLOANE)
                                            break
                            print("\nTabla dupa mutarea jucatorului")
                            print(str(stare_curenta))

                    if pos_start[0] > pos_end[0] and pos_start[1] < pos_end[1]:
                        test1 = 0
                        for np1 in range(len(Joc.celuleGrid)):
                            if Joc.celuleGrid[np1].collidepoint((pos_end[0], pos_start[1])) and test1 == 0:
                                for np2 in range(len(Joc.celuleGrid)):
                                    if Joc.celuleGrid[np2].collidepoint((pos_start[0], pos_end[1])) and test1 == 0:
                                        for g in range(np1, np2 + 1):
                                            if (g // Joc.NR_COLOANE <= np2 // Joc.NR_COLOANE and g % Joc.NR_COLOANE <= np2 % Joc.NR_COLOANE) and (
                                                    g // Joc.NR_COLOANE >= np1 // Joc.NR_COLOANE and g % Joc.NR_COLOANE >= np1 % Joc.NR_COLOANE) and (
                                                    (stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][
                                                        g % Joc.NR_COLOANE] == 'r')  or (stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][g % Joc.NR_COLOANE] == '0') or (stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][g % Joc.NR_COLOANE] == 'x')):
                                                test1 = 1
                                                print("Nu puteti manca zona aleasa deoarece contine cel putin o bucata otravita sau aleasa deja!")
                                                break

                        if test1 == 0:
                            nri = 0
                            for np1 in range(len(Joc.celuleGrid)):
                                if Joc.celuleGrid[np1].collidepoint((pos_end[0], pos_start[1])) and test1 == 0:
                                    np = np1
                                    su = 0
                                    st = 0
                                    if np1 >= Joc.NR_COLOANE:
                                        su = 1
                                        np = np - Joc.NR_COLOANE
                                    if np1 % Joc.NR_COLOANE != 0:
                                        st = 1
                                        np = np - 1
                                    np1 = np
                                    i1 = 0
                                    i2 = Joc.NR_LINII + Joc.NR_COLOANE + 5
                                    # print(str(i1) + " " + str(i2))
                                    for np2 in range(len(Joc.celuleGrid)):
                                        if Joc.celuleGrid[np2].collidepoint(pos_start[0], pos_end[1]) and test1 == 0:
                                            np = np2
                                            jo = 0
                                            dr = 0
                                            if np2 <= (Joc.NR_COLOANE * Joc.NR_LINII) - Joc.NR_COLOANE - 1:
                                                jo = 1
                                                np = np + Joc.NR_COLOANE
                                            if (np2 + 1) % Joc.NR_COLOANE != 0:
                                                dr = 1
                                                np = np + 1
                                            np2 = np
                                            i3 = Joc.NR_LINII + Joc.NR_COLOANE * 2 + 8
                                            i4 = Joc.NR_LINII + Joc.NR_COLOANE * 2 + 9
                                            listm = [-1 for _ in range((Joc.NR_COLOANE + Joc.NR_LINII + 20) * 2 + 1)]
                                            for g in range(np1, np2 + 1):
                                                if (
                                                        g // Joc.NR_COLOANE == np2 // Joc.NR_COLOANE and g % Joc.NR_COLOANE <= np2 % Joc.NR_COLOANE) and (
                                                        g // Joc.NR_COLOANE >= np1 // Joc.NR_COLOANE and g % Joc.NR_COLOANE >= np1 % Joc.NR_COLOANE):
                                                    if jo == 1:

                                                        if stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][g % Joc.NR_COLOANE] == 'x' or stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][g % Joc.NR_COLOANE] == '0':
                                                            listm[i1] = 'x'  # 'x'
                                                        else:
                                                            listm[i1] = '#'
                                                        i1 += 1
                                                    else:
                                                        listm[i1] = 'x'
                                                        i1 += 1

                                                if (
                                                        g // Joc.NR_COLOANE <= np2 // Joc.NR_COLOANE and g % Joc.NR_COLOANE == np2 % Joc.NR_COLOANE) and (
                                                        g // Joc.NR_COLOANE >= np1 // Joc.NR_COLOANE and g % Joc.NR_COLOANE >= np1 % Joc.NR_COLOANE):
                                                    if dr == 1:


                                                        if stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][g % Joc.NR_COLOANE] == 'x' or stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][g % Joc.NR_COLOANE] == '0':
                                                            listm[i2] = 'x'
                                                        else:
                                                            listm[i2] = '#'
                                                        i2 -= 1
                                                    else:
                                                        listm[i2] = 'x'
                                                        i2 -= 1

                                                if (
                                                        g // Joc.NR_COLOANE <= np2 // Joc.NR_COLOANE and g % Joc.NR_COLOANE <= np2 % Joc.NR_COLOANE) and (
                                                        g // Joc.NR_COLOANE == np1 // Joc.NR_COLOANE and g % Joc.NR_COLOANE >= np1 % Joc.NR_COLOANE):
                                                    if su == 1:

                                                        if stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][g % Joc.NR_COLOANE] == 'x' or stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][g % Joc.NR_COLOANE] == '0':
                                                            listm[i3] = 'x'
                                                        else:
                                                            listm[i3] = '#'
                                                        i3 -= 1
                                                    else:
                                                        listm[i3] = 'x'
                                                        i3 -= 1

                                                if (
                                                        g // Joc.NR_COLOANE <= np2 // Joc.NR_COLOANE and g % Joc.NR_COLOANE <= np2 % Joc.NR_COLOANE) and (
                                                        g // Joc.NR_COLOANE >= np1 // Joc.NR_COLOANE and g % Joc.NR_COLOANE == np1 % Joc.NR_COLOANE):
                                                    if st == 1:

                                                        if stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][g % Joc.NR_COLOANE] == 'x' or stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][g % Joc.NR_COLOANE] == '0':
                                                            listm[i4] = 'x'
                                                        else:
                                                            listm[i4] = '#'
                                                        i4 += 1
                                                    else:
                                                        listm[i4] = 'x'
                                                        i4 += 1

                            listm = list(filter(lambda a: a != -1, listm))
                            listm.append(listm[0])
                            # print(listm)
                            for _ in range(len(listm) - 1):
                                # print(_)
                                if (listm[_] == 'x' and listm[_ + 1] == '#') or (listm[_] == '#' and listm[_ + 1] == 'x'):
                                    nri += 1
                            if nri > 2 or nri <= 0:
                                test1 = 1

                        if test1 == 0:
                            for np1 in range(len(Joc.celuleGrid)):
                                if Joc.celuleGrid[np1].collidepoint((pos_end[0], pos_start[1])):
                                    for np2 in range(len(Joc.celuleGrid)):
                                        if Joc.celuleGrid[np2].collidepoint((pos_start[0], pos_end[1])):
                                            for g in range(np1, np2 + 1):
                                                if (g // Joc.NR_COLOANE <= np2 // Joc.NR_COLOANE and g % Joc.NR_COLOANE <= np2 % Joc.NR_COLOANE) and (g // Joc.NR_COLOANE >= np1 // Joc.NR_COLOANE and g % Joc.NR_COLOANE >= np1 % Joc.NR_COLOANE):
                                                    stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][g % Joc.NR_COLOANE] = Joc.JMIN
                                                    stare_curenta.tabla_joc.ultima_mutare = (g - np1, g % Joc.NR_COLOANE)
                                                    stare_curenta.tabla_joc.deseneaza_grid(linie_marcaj=g // Joc.NR_COLOANE,
                                                                                   coloana_marcaj=g % Joc.NR_COLOANE)
                                            break
                            print("\nTabla dupa mutarea jucatorului")
                            print(str(stare_curenta))



                    if pos_start[0] > pos_end[0] and pos_start[1] > pos_end[1]:
                        test1 = 0
                        for np1 in range(len(Joc.celuleGrid)):
                            if Joc.celuleGrid[np1].collidepoint((pos_end[0], pos_end[1])) and test1 == 0:
                                for np2 in range(len(Joc.celuleGrid)):
                                    if Joc.celuleGrid[np2].collidepoint((pos_start[0], pos_start[1])) and test1 == 0:
                                        for g in range(np1, np2 + 1):
                                            if (g // Joc.NR_COLOANE <= np2 // Joc.NR_COLOANE and g % Joc.NR_COLOANE <= np2 % Joc.NR_COLOANE) and (
                                                    g // Joc.NR_COLOANE >= np1 // Joc.NR_COLOANE and g % Joc.NR_COLOANE >= np1 % Joc.NR_COLOANE) and (
                                                    (stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][g % Joc.NR_COLOANE] == 'r')  or (stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][g % Joc.NR_COLOANE] == '0') or (stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][g % Joc.NR_COLOANE] == 'x')):
                                                test1 = 1
                                                print("Nu puteti manca zona aleasa deoarece contine cel putin o bucata otravita sau aleasa deja!")
                                                break

                        if test1 == 0:
                            nri = 0
                            for np1 in range(len(Joc.celuleGrid)):
                                if Joc.celuleGrid[np1].collidepoint((pos_end[0], pos_end[1])) and test1 == 0:
                                    np = np1
                                    su = 0
                                    st = 0
                                    if np1 >= Joc.NR_COLOANE:
                                        su = 1
                                        np = np - Joc.NR_COLOANE
                                    if np1 % Joc.NR_COLOANE != 0:
                                        st = 1
                                        np = np - 1
                                    np1 = np
                                    i1 = 0
                                    i2 = Joc.NR_LINII + Joc.NR_COLOANE + 5
                                    # print(str(i1) + " " + str(i2))
                                    for np2 in range(len(Joc.celuleGrid)):
                                        if Joc.celuleGrid[np2].collidepoint(pos_start[0], pos_start[1]) and test1 == 0:
                                            np = np2
                                            jo = 0
                                            dr = 0
                                            if np2 <= (Joc.NR_COLOANE * Joc.NR_LINII) - Joc.NR_COLOANE - 1:
                                                jo = 1
                                                np = np + Joc.NR_COLOANE
                                            if (np2 + 1) % Joc.NR_COLOANE != 0:
                                                dr = 1
                                                np = np + 1
                                            np2 = np
                                            i3 = Joc.NR_LINII + Joc.NR_COLOANE * 2 + 8
                                            i4 = Joc.NR_LINII + Joc.NR_COLOANE * 2 + 9
                                            listm = [-1 for _ in range((Joc.NR_COLOANE + Joc.NR_LINII + 20) * 2 + 1)]
                                            for g in range(np1, np2 + 1):
                                                if (
                                                        g // Joc.NR_COLOANE == np2 // Joc.NR_COLOANE and g % Joc.NR_COLOANE <= np2 % Joc.NR_COLOANE) and (
                                                        g // Joc.NR_COLOANE >= np1 // Joc.NR_COLOANE and g % Joc.NR_COLOANE >= np1 % Joc.NR_COLOANE):
                                                    if jo == 1:
                                                        stare_curenta.tabla_joc.deseneaza_grid(
                                                                linie_marcaj=g // Joc.NR_COLOANE,
                                                                coloana_marcaj=g % Joc.NR_COLOANE)
                                                        if stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][g % Joc.NR_COLOANE] == 'x' or stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][g % Joc.NR_COLOANE] == '0':
                                                            listm[i1] = 'x'  # 'x'
                                                        else:
                                                            listm[i1] = '#'
                                                        i1 += 1
                                                    else:
                                                        listm[i1] = 'x'
                                                        i1 += 1

                                                if (
                                                        g // Joc.NR_COLOANE <= np2 // Joc.NR_COLOANE and g % Joc.NR_COLOANE == np2 % Joc.NR_COLOANE) and (
                                                        g // Joc.NR_COLOANE >= np1 // Joc.NR_COLOANE and g % Joc.NR_COLOANE >= np1 % Joc.NR_COLOANE):
                                                    if dr == 1:

                                                        stare_curenta.tabla_joc.deseneaza_grid(
                                                                linie_marcaj=g // Joc.NR_COLOANE,
                                                                coloana_marcaj=g % Joc.NR_COLOANE)
                                                        if stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][g % Joc.NR_COLOANE] == 'x' or stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][g % Joc.NR_COLOANE] == '0':
                                                            listm[i2] = 'x'
                                                        else:
                                                            listm[i2] = '#'
                                                        i2 -= 1
                                                    else:
                                                        listm[i2] = 'x'
                                                        i2 -= 1

                                                if (
                                                        g // Joc.NR_COLOANE <= np2 // Joc.NR_COLOANE and g % Joc.NR_COLOANE <= np2 % Joc.NR_COLOANE) and (
                                                        g // Joc.NR_COLOANE == np1 // Joc.NR_COLOANE and g % Joc.NR_COLOANE >= np1 % Joc.NR_COLOANE):
                                                    if su == 1:
                                                        stare_curenta.tabla_joc.deseneaza_grid(
                                                                linie_marcaj=g // Joc.NR_COLOANE,
                                                                coloana_marcaj=g % Joc.NR_COLOANE)
                                                        if stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][g % Joc.NR_COLOANE] == 'x' or stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][g % Joc.NR_COLOANE] == '0':
                                                            listm[i3] = 'x'
                                                        else:
                                                            listm[i3] = '#'
                                                        i3 -= 1
                                                    else:
                                                        listm[i3] = 'x'
                                                        i3 -= 1

                                                if (
                                                        g // Joc.NR_COLOANE <= np2 // Joc.NR_COLOANE and g % Joc.NR_COLOANE <= np2 % Joc.NR_COLOANE) and (
                                                        g // Joc.NR_COLOANE >= np1 // Joc.NR_COLOANE and g % Joc.NR_COLOANE == np1 % Joc.NR_COLOANE):
                                                    if st == 1:
                                                        stare_curenta.tabla_joc.deseneaza_grid(
                                                                linie_marcaj=g // Joc.NR_COLOANE,
                                                                coloana_marcaj=g % Joc.NR_COLOANE)
                                                        if stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][g % Joc.NR_COLOANE] == 'x' or stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][g % Joc.NR_COLOANE] == '0':
                                                            listm[i4] = 'x'
                                                        else:
                                                            listm[i4] = '#'
                                                        i4 += 1
                                                    else:
                                                        listm[i4] = 'x'
                                                        i4 += 1

                            listm = list(filter(lambda a: a != -1, listm))
                            listm.append(listm[0])
                            # print(listm)
                            for _ in range(len(listm) - 1):
                                # print(_)
                                if (listm[_] == 'x' and listm[_ + 1] == '#') or (listm[_] == '#' and listm[_ + 1] == 'x'):
                                    nri += 1
                            if nri > 2 or nri <= 0:
                                test1 = 1

                        if test1 == 0:
                            for np1 in range(len(Joc.celuleGrid)):
                                if Joc.celuleGrid[np1].collidepoint((pos_end[0], pos_end[1])):
                                    for np2 in range(len(Joc.celuleGrid)):
                                        if Joc.celuleGrid[np2].collidepoint((pos_start[0], pos_start[1])):
                                            for g in range(np1, np2 + 1):
                                                if (g // Joc.NR_COLOANE <= np2 // Joc.NR_COLOANE and g % Joc.NR_COLOANE <= np2 % Joc.NR_COLOANE) and (g // Joc.NR_COLOANE >= np1 // Joc.NR_COLOANE and g % Joc.NR_COLOANE >= np1 % Joc.NR_COLOANE):
                                                    stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][g % Joc.NR_COLOANE] = Joc.JMIN
                                                    stare_curenta.tabla_joc.ultima_mutare = (g - np1, g % Joc.NR_COLOANE)
                                                    # print(str(stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][g % Joc.NR_COLOANE]))
                                                    stare_curenta.tabla_joc.deseneaza_grid(linie_marcaj=g // Joc.NR_COLOANE,
                                                                                   coloana_marcaj=g % Joc.NR_COLOANE)
                                            break
                            print("\nTabla dupa mutarea jucatorului")
                            print(str(stare_curenta))

                    pygame.display.flip()
                    if (afis_daca_final(stare_curenta)):
                        break
                    # daca test1 este 0, acest 'if' va trece la randul celuilalt jucator
                    if test1 == 0:
                        linie = np // Joc.NR_COLOANE
                        coloana = np % Joc.NR_COLOANE

                        # 9.a) -afisarea timpului de gandire
                        # afisul timpului in care jucatorul si-a facut mutarea
                        t_dupaom = int(round(time.time() * 1000))
                        print("Jucatorul a \"gandit\" timp de " + str(t_dupaom - t_inainteom) + " milisecunde.")
                        test_timp = 0
                        # afisarea starii jocului in urma mutarii utilizatorului
                        print("\nTabla dupa mutarea jucatorului")
                        print(str(stare_curenta))

                        stare_curenta.tabla_joc.deseneaza_grid(linie_marcaj=linie, coloana_marcaj=coloana)
                        # testez daca jocul a ajuns intr-o stare finala
                        # si afisez un mesaj corespunzator in caz ca da
                        Joc.jucator_curent = '0'
                        if (afis_daca_final(stare_curenta)):
                            break
                        # trecerea efectiva la celalalt jucator
                        stare_curenta.j_curent = Joc.jucator_opus(stare_curenta.j_curent)

        # --------------------------------
        else:  # jucatorul e JMAX (calculatorul)
            # Mutare calculator daca tip_joc este 2, AI vs AI daca tip_juc este 1 si PvP daca tip_juc este 3
            # 13. - optiuni jucatori
            if Joc.tip_joc == 2:
                # preiau timpul in milisecunde de dinainte de mutare
                # 9.a) -afisarea timpului de gandire
                t_inainte = int(round(time.time() * 1000))

                if tip_algoritm == 'minimax':
                    stare_actualizata = min_max(stare_curenta)
                else:  # tip_algoritm=="alphabeta"
                    stare_actualizata = alpha_beta(-500, 500, stare_curenta)
                stare_curenta.tabla_joc = stare_actualizata.stare_aleasa.tabla_joc

                print("Tabla dupa mutarea calculatorului\n" + str(stare_curenta))

                # 9.a) -afisarea timpului de gandire
                # preiau timpul in milisecunde de dupa mutare
                t_dupa = int(round(time.time() * 1000))
                print("Calculatorul a \"gandit\" timp de " + str(t_dupa - t_inainte) + " milisecunde.")

                stare_curenta.tabla_joc.deseneaza_grid()
                if (afis_daca_final(stare_curenta)):
                    break

                # S-a realizat o mutare. Schimb jucatorul cu cel opus
                stare_curenta.j_curent = Joc.jucator_opus(stare_curenta.j_curent)

            # 13. - optiuni jucatori
            elif int(Joc.tip_joc) == 3:
            # functie analog celei pentru celalalt jucator
                for event in pygame.event.get():

                    if event.type == pygame.QUIT:
                        # iesim din program
                        pygame.quit()
                        sys.exit()
                    if event.type == pygame.MOUSEMOTION:

                        pos = pygame.mouse.get_pos()  # coordonatele cursorului
                        for np in range(len(Joc.celuleGrid)):
                            if Joc.celuleGrid[np].collidepoint(pos):
                                stare_curenta.tabla_joc.deseneaza_grid(linie_marcaj=np // Joc.NR_COLOANE,
                                                                       coloana_marcaj=np % Joc.NR_COLOANE)
                                break

                    elif event.type == pygame.MOUSEBUTTONDOWN and selectat == 0:
                        selectat = 1
                        pos_start = pygame.mouse.get_pos()  # coordonatele cursorului la momentul clickului

                    elif event.type == pygame.MOUSEBUTTONDOWN and selectat == 1:
                        selectat = 0
                        pos_end = pygame.mouse.get_pos()  # coordonatele cursorului la momentul eliberarii clickului

                        # cazurile in functie de modul in care jucatorul a ales zona
                        if pos_start[0] <= pos_end[0] and pos_start[1] <= pos_end[1]:
                            test1 = 0
                            for np1 in range(len(Joc.celuleGrid)):
                                if Joc.celuleGrid[np1].collidepoint((pos_start[0], pos_start[1])) and test1 == 0:
                                    for np2 in range(len(Joc.celuleGrid)):
                                        if Joc.celuleGrid[np2].collidepoint(pos_end) and test1 == 0:
                                            for g in range(np1, np2 + 1):
                                                # print(str(stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][g % Joc.NR_COLOANE]))
                                                if (
                                                        g // Joc.NR_COLOANE <= np2 // Joc.NR_COLOANE and g % Joc.NR_COLOANE <= np2 % Joc.NR_COLOANE) and (
                                                        g // Joc.NR_COLOANE >= np1 // Joc.NR_COLOANE and g % Joc.NR_COLOANE >= np1 % Joc.NR_COLOANE) and (
                                                        (stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][
                                                             g % Joc.NR_COLOANE] == 'r') or (
                                                                stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][
                                                                    g % Joc.NR_COLOANE] == '0') or (
                                                                stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][
                                                                    g % Joc.NR_COLOANE] == 'x')):
                                                    test1 = 1
                                                    print(
                                                        "Nu puteti manca zona aleasa deoarece contine cel putin o bucata otravita sau aleasa deja!")
                                                    break
                            if test1 == 0:
                                nri = 0
                                for np1 in range(len(Joc.celuleGrid)):
                                    if Joc.celuleGrid[np1].collidepoint((pos_start[0], pos_start[1])) and test1 == 0:
                                        np = np1
                                        su = 0
                                        st = 0
                                        if np1 >= Joc.NR_COLOANE:
                                            su = 1
                                            np = np - Joc.NR_COLOANE
                                        if np1 % Joc.NR_COLOANE != 0:
                                            st = 1
                                            np = np - 1
                                        np1 = np
                                        i1 = 0
                                        i2 = Joc.NR_LINII + Joc.NR_COLOANE + 5
                                        # print(str(i1) + " " + str(i2))
                                        for np2 in range(len(Joc.celuleGrid)):
                                            if Joc.celuleGrid[np2].collidepoint(pos_end) and test1 == 0:
                                                np = np2
                                                jo = 0
                                                dr = 0
                                                if np2 <= (Joc.NR_COLOANE * Joc.NR_LINII) - Joc.NR_COLOANE - 1:
                                                    jo = 1
                                                    np = np + Joc.NR_COLOANE
                                                if (np2 + 1) % Joc.NR_COLOANE != 0:
                                                    dr = 1
                                                    np = np + 1
                                                np2 = np
                                                i3 = Joc.NR_LINII + Joc.NR_COLOANE * 2 + 8
                                                i4 = Joc.NR_LINII + Joc.NR_COLOANE * 2 + 9
                                                listm = [-1 for _ in
                                                         range((Joc.NR_COLOANE + Joc.NR_LINII + 20) * 2 + 1)]
                                                for g in range(np1, np2 + 1):
                                                    if (
                                                            g // Joc.NR_COLOANE == np2 // Joc.NR_COLOANE and g % Joc.NR_COLOANE <= np2 % Joc.NR_COLOANE) and (
                                                            g // Joc.NR_COLOANE >= np1 // Joc.NR_COLOANE and g % Joc.NR_COLOANE >= np1 % Joc.NR_COLOANE):
                                                        if jo == 1:
                                                            stare_curenta.tabla_joc.deseneaza_grid(
                                                                linie_marcaj=g // Joc.NR_COLOANE,
                                                                coloana_marcaj=g % Joc.NR_COLOANE)
                                                            if stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][
                                                                g % Joc.NR_COLOANE] == 'x' or \
                                                                    stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][
                                                                        g % Joc.NR_COLOANE] == '0':
                                                                listm[i1] = 'x'  # 'x'
                                                            else:
                                                                listm[i1] = '#'
                                                            i1 += 1
                                                        else:
                                                            listm[i1] = 'x'
                                                            i1 += 1

                                                    if (
                                                            g // Joc.NR_COLOANE <= np2 // Joc.NR_COLOANE and g % Joc.NR_COLOANE == np2 % Joc.NR_COLOANE) and (
                                                            g // Joc.NR_COLOANE >= np1 // Joc.NR_COLOANE and g % Joc.NR_COLOANE >= np1 % Joc.NR_COLOANE):
                                                        if dr == 1:

                                                            stare_curenta.tabla_joc.deseneaza_grid(
                                                                linie_marcaj=g // Joc.NR_COLOANE,
                                                                coloana_marcaj=g % Joc.NR_COLOANE)
                                                            if stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][
                                                                g % Joc.NR_COLOANE] == 'x' or \
                                                                    stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][
                                                                        g % Joc.NR_COLOANE] == '0':
                                                                listm[i2] = 'x'
                                                            else:
                                                                listm[i2] = '#'
                                                            i2 -= 1
                                                        else:
                                                            listm[i2] = 'x'
                                                            i2 -= 1

                                                    if (
                                                            g // Joc.NR_COLOANE <= np2 // Joc.NR_COLOANE and g % Joc.NR_COLOANE <= np2 % Joc.NR_COLOANE) and (
                                                            g // Joc.NR_COLOANE == np1 // Joc.NR_COLOANE and g % Joc.NR_COLOANE >= np1 % Joc.NR_COLOANE):
                                                        if su == 1:
                                                            stare_curenta.tabla_joc.deseneaza_grid(
                                                                linie_marcaj=g // Joc.NR_COLOANE,
                                                                coloana_marcaj=g % Joc.NR_COLOANE)
                                                            if stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][
                                                                g % Joc.NR_COLOANE] == 'x' or \
                                                                    stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][
                                                                        g % Joc.NR_COLOANE] == '0':
                                                                listm[i3] = 'x'
                                                            else:
                                                                listm[i3] = '#'
                                                            i3 -= 1
                                                        else:
                                                            listm[i3] = 'x'
                                                            i3 -= 1

                                                    if (
                                                            g // Joc.NR_COLOANE <= np2 // Joc.NR_COLOANE and g % Joc.NR_COLOANE <= np2 % Joc.NR_COLOANE) and (
                                                            g // Joc.NR_COLOANE >= np1 // Joc.NR_COLOANE and g % Joc.NR_COLOANE == np1 % Joc.NR_COLOANE):
                                                        if st == 1:
                                                            stare_curenta.tabla_joc.deseneaza_grid(
                                                                linie_marcaj=g // Joc.NR_COLOANE,
                                                                coloana_marcaj=g % Joc.NR_COLOANE)
                                                            if stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][
                                                                g % Joc.NR_COLOANE] == 'x' or \
                                                                    stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][
                                                                        g % Joc.NR_COLOANE] == '0':
                                                                listm[i4] = 'x'
                                                            else:
                                                                listm[i4] = '#'
                                                            i4 += 1
                                                        else:
                                                            listm[i4] = 'x'
                                                            i4 += 1

                                listm = list(filter(lambda a: a != -1, listm))
                                listm.append(listm[0])
                                # print(listm)
                                for _ in range(len(listm) - 1):
                                    # print(_)
                                    if (listm[_] == 'x' and listm[_ + 1] == '#') or (
                                            listm[_] == '#' and listm[_ + 1] == 'x'):
                                        nri += 1
                                if nri > 2 or nri <= 0:
                                    test1 = 1

                            if test1 == 0:
                                for np1 in range(len(Joc.celuleGrid)):
                                    if Joc.celuleGrid[np1].collidepoint((pos_start[0], pos_start[1])):
                                        for np2 in range(len(Joc.celuleGrid)):
                                            if Joc.celuleGrid[np2].collidepoint(pos_end):
                                                for g in range(np1, np2 + 1):
                                                    if (
                                                            g // Joc.NR_COLOANE <= np2 // Joc.NR_COLOANE and g % Joc.NR_COLOANE <= np2 % Joc.NR_COLOANE) and (
                                                            g // Joc.NR_COLOANE >= np1 // Joc.NR_COLOANE and g % Joc.NR_COLOANE >= np1 % Joc.NR_COLOANE):
                                                        print("AIIIIIIIIIICICICI: ")
                                                        print(Joc.JMAX)
                                                        stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][
                                                            g % Joc.NR_COLOANE] = Joc.JMAX
                                                        stare_curenta.tabla_joc.ultima_mutare = (
                                                        g - np1, g % Joc.NR_COLOANE)
                                                        # print(str(stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][g % Joc.NR_COLOANE]))
                                                        stare_curenta.tabla_joc.deseneaza_grid(
                                                            linie_marcaj=g // Joc.NR_COLOANE,
                                                            coloana_marcaj=g % Joc.NR_COLOANE)
                                                break
                                print("\nTabla dupa mutarea jucatorului")
                                print(str(stare_curenta))

                        if pos_start[0] < pos_end[0] and pos_start[1] > pos_end[1]:
                            test1 = 0
                            for np1 in range(len(Joc.celuleGrid)):
                                if Joc.celuleGrid[np1].collidepoint((pos_start[0], pos_end[1])) and test1 == 0:
                                    for np2 in range(len(Joc.celuleGrid)):
                                        if Joc.celuleGrid[np2].collidepoint((pos_end[0], pos_start[1])) and test1 == 0:
                                            for g in range(np1, np2 + 1):
                                                if (
                                                        g // Joc.NR_COLOANE <= np2 // Joc.NR_COLOANE and g % Joc.NR_COLOANE <= np2 % Joc.NR_COLOANE) and (
                                                        g // Joc.NR_COLOANE >= np1 // Joc.NR_COLOANE and g % Joc.NR_COLOANE >= np1 % Joc.NR_COLOANE) and (
                                                        (stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][
                                                             g % Joc.NR_COLOANE] == 'r') or (
                                                                stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][
                                                                    g % Joc.NR_COLOANE] == '0') or (
                                                                stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][
                                                                    g % Joc.NR_COLOANE] == 'x')):
                                                    test1 = 1
                                                    print(
                                                        "Nu puteti manca zona aleasa deoarece contine cel putin o bucata otravita sau aleasa deja!")
                                                    break

                            if test1 == 0:
                                nri = 0
                                for np1 in range(len(Joc.celuleGrid)):
                                    if Joc.celuleGrid[np1].collidepoint((pos_start[0], pos_end[1])) and test1 == 0:
                                        np = np1
                                        su = 0
                                        st = 0
                                        if np1 >= Joc.NR_COLOANE:
                                            su = 1
                                            np = np - Joc.NR_COLOANE
                                        if np1 % Joc.NR_COLOANE != 0:
                                            st = 1
                                            np = np - 1
                                        np1 = np
                                        i1 = 0
                                        i2 = Joc.NR_LINII + Joc.NR_COLOANE + 5
                                        # print(str(i1) + " " + str(i2))
                                        for np2 in range(len(Joc.celuleGrid)):
                                            if Joc.celuleGrid[np2].collidepoint(pos_end[0],
                                                                                pos_start[1]) and test1 == 0:
                                                np = np2
                                                jo = 0
                                                dr = 0
                                                if np2 <= (Joc.NR_COLOANE * Joc.NR_LINII) - Joc.NR_COLOANE - 1:
                                                    jo = 1
                                                    np = np + Joc.NR_COLOANE
                                                if (np2 + 1) % Joc.NR_COLOANE != 0:
                                                    dr = 1
                                                    np = np + 1
                                                np2 = np
                                                i3 = Joc.NR_LINII + Joc.NR_COLOANE * 2 + 8
                                                i4 = Joc.NR_LINII + Joc.NR_COLOANE * 2 + 9
                                                listm = [-1 for _ in
                                                         range((Joc.NR_COLOANE + Joc.NR_LINII + 20) * 2 + 1)]
                                                for g in range(np1, np2 + 1):
                                                    if (
                                                            g // Joc.NR_COLOANE == np2 // Joc.NR_COLOANE and g % Joc.NR_COLOANE <= np2 % Joc.NR_COLOANE) and (
                                                            g // Joc.NR_COLOANE >= np1 // Joc.NR_COLOANE and g % Joc.NR_COLOANE >= np1 % Joc.NR_COLOANE):
                                                        if jo == 1:
                                                            stare_curenta.tabla_joc.deseneaza_grid(
                                                                linie_marcaj=g // Joc.NR_COLOANE,
                                                                coloana_marcaj=g % Joc.NR_COLOANE)
                                                            if stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][
                                                                g % Joc.NR_COLOANE] == 'x' or \
                                                                    stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][
                                                                        g % Joc.NR_COLOANE] == '0':
                                                                listm[i1] = 'x'  # 'x'
                                                            else:
                                                                listm[i1] = '#'
                                                            i1 += 1
                                                        else:
                                                            listm[i1] = 'x'
                                                            i1 += 1

                                                    if (
                                                            g // Joc.NR_COLOANE <= np2 // Joc.NR_COLOANE and g % Joc.NR_COLOANE == np2 % Joc.NR_COLOANE) and (
                                                            g // Joc.NR_COLOANE >= np1 // Joc.NR_COLOANE and g % Joc.NR_COLOANE >= np1 % Joc.NR_COLOANE):
                                                        if dr == 1:

                                                            stare_curenta.tabla_joc.deseneaza_grid(
                                                                linie_marcaj=g // Joc.NR_COLOANE,
                                                                coloana_marcaj=g % Joc.NR_COLOANE)
                                                            if stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][
                                                                g % Joc.NR_COLOANE] == 'x' or \
                                                                    stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][
                                                                        g % Joc.NR_COLOANE] == '0':
                                                                listm[i2] = 'x'
                                                            else:
                                                                listm[i2] = '#'
                                                            i2 -= 1
                                                        else:
                                                            listm[i2] = 'x'
                                                            i2 -= 1

                                                    if (
                                                            g // Joc.NR_COLOANE <= np2 // Joc.NR_COLOANE and g % Joc.NR_COLOANE <= np2 % Joc.NR_COLOANE) and (
                                                            g // Joc.NR_COLOANE == np1 // Joc.NR_COLOANE and g % Joc.NR_COLOANE >= np1 % Joc.NR_COLOANE):
                                                        if su == 1:
                                                            stare_curenta.tabla_joc.deseneaza_grid(
                                                                linie_marcaj=g // Joc.NR_COLOANE,
                                                                coloana_marcaj=g % Joc.NR_COLOANE)
                                                            if stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][
                                                                g % Joc.NR_COLOANE] == 'x' or \
                                                                    stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][
                                                                        g % Joc.NR_COLOANE] == '0':
                                                                listm[i3] = 'x'
                                                            else:
                                                                listm[i3] = '#'
                                                            i3 -= 1
                                                        else:
                                                            listm[i3] = 'x'
                                                            i3 -= 1

                                                    if (
                                                            g // Joc.NR_COLOANE <= np2 // Joc.NR_COLOANE and g % Joc.NR_COLOANE <= np2 % Joc.NR_COLOANE) and (
                                                            g // Joc.NR_COLOANE >= np1 // Joc.NR_COLOANE and g % Joc.NR_COLOANE == np1 % Joc.NR_COLOANE):
                                                        if st == 1:
                                                            stare_curenta.tabla_joc.deseneaza_grid(
                                                                linie_marcaj=g // Joc.NR_COLOANE,
                                                                coloana_marcaj=g % Joc.NR_COLOANE)
                                                            if stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][
                                                                g % Joc.NR_COLOANE] == 'x' or \
                                                                    stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][
                                                                        g % Joc.NR_COLOANE] == '0':
                                                                listm[i4] = 'x'
                                                            else:
                                                                listm[i4] = '#'
                                                            i4 += 1
                                                        else:
                                                            listm[i4] = 'x'
                                                            i4 += 1

                                listm = list(filter(lambda a: a != -1, listm))
                                listm.append(listm[0])
                                # print(listm)
                                for _ in range(len(listm) - 1):
                                    # print(_)
                                    if (listm[_] == 'x' and listm[_ + 1] == '#') or (
                                            listm[_] == '#' and listm[_ + 1] == 'x'):
                                        nri += 1
                                if nri > 2 or nri <= 0:
                                    test1 = 1

                            if test1 == 0:
                                for np1 in range(len(Joc.celuleGrid)):
                                    if Joc.celuleGrid[np1].collidepoint((pos_start[0], pos_end[1])):
                                        for np2 in range(len(Joc.celuleGrid)):
                                            if Joc.celuleGrid[np2].collidepoint((pos_end[0], pos_start[1])):
                                                for g in range(np1, np2 + 1):
                                                    if (
                                                            g // Joc.NR_COLOANE <= np2 // Joc.NR_COLOANE and g % Joc.NR_COLOANE <= np2 % Joc.NR_COLOANE) and (
                                                            g // Joc.NR_COLOANE >= np1 // Joc.NR_COLOANE and g % Joc.NR_COLOANE >= np1 % Joc.NR_COLOANE):
                                                        stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][
                                                            g % Joc.NR_COLOANE] = Joc.JMAX
                                                        stare_curenta.tabla_joc.ultima_mutare = (
                                                        g - np1, g % Joc.NR_COLOANE)
                                                        # print(str(stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][g % Joc.NR_COLOANE]))
                                                        stare_curenta.tabla_joc.deseneaza_grid(
                                                            linie_marcaj=g // Joc.NR_COLOANE,
                                                            coloana_marcaj=g % Joc.NR_COLOANE)
                                                break
                                print("\nTabla dupa mutarea jucatorului")
                                print(str(stare_curenta))

                        if pos_start[0] > pos_end[0] and pos_start[1] < pos_end[1]:
                            test1 = 0
                            for np1 in range(len(Joc.celuleGrid)):
                                if Joc.celuleGrid[np1].collidepoint((pos_end[0], pos_start[1])) and test1 == 0:
                                    for np2 in range(len(Joc.celuleGrid)):
                                        if Joc.celuleGrid[np2].collidepoint((pos_start[0], pos_end[1])) and test1 == 0:
                                            for g in range(np1, np2 + 1):
                                                if (
                                                        g // Joc.NR_COLOANE <= np2 // Joc.NR_COLOANE and g % Joc.NR_COLOANE <= np2 % Joc.NR_COLOANE) and (
                                                        g // Joc.NR_COLOANE >= np1 // Joc.NR_COLOANE and g % Joc.NR_COLOANE >= np1 % Joc.NR_COLOANE) and (
                                                        (stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][
                                                             g % Joc.NR_COLOANE] == 'r') or (
                                                                stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][
                                                                    g % Joc.NR_COLOANE] == '0') or (
                                                                stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][
                                                                    g % Joc.NR_COLOANE] == 'x')):
                                                    test1 = 1
                                                    print(
                                                        "Nu puteti manca zona aleasa deoarece contine cel putin o bucata otravita sau aleasa deja!")
                                                    break

                            if test1 == 0:
                                nri = 0
                                for np1 in range(len(Joc.celuleGrid)):
                                    if Joc.celuleGrid[np1].collidepoint((pos_end[0], pos_start[1])) and test1 == 0:
                                        np = np1
                                        su = 0
                                        st = 0
                                        if np1 >= Joc.NR_COLOANE:
                                            su = 1
                                            np = np - Joc.NR_COLOANE
                                        if np1 % Joc.NR_COLOANE != 0:
                                            st = 1
                                            np = np - 1
                                        np1 = np
                                        i1 = 0
                                        i2 = Joc.NR_LINII + Joc.NR_COLOANE + 5
                                        # print(str(i1) + " " + str(i2))
                                        for np2 in range(len(Joc.celuleGrid)):
                                            if Joc.celuleGrid[np2].collidepoint(pos_start[0],
                                                                                pos_end[1]) and test1 == 0:
                                                np = np2
                                                jo = 0
                                                dr = 0
                                                if np2 <= (Joc.NR_COLOANE * Joc.NR_LINII) - Joc.NR_COLOANE - 1:
                                                    jo = 1
                                                    np = np + Joc.NR_COLOANE
                                                if (np2 + 1) % Joc.NR_COLOANE != 0:
                                                    dr = 1
                                                    np = np + 1
                                                np2 = np
                                                i3 = Joc.NR_LINII + Joc.NR_COLOANE * 2 + 8
                                                i4 = Joc.NR_LINII + Joc.NR_COLOANE * 2 + 9
                                                listm = [-1 for _ in
                                                         range((Joc.NR_COLOANE + Joc.NR_LINII + 20) * 2 + 1)]
                                                for g in range(np1, np2 + 1):
                                                    if (
                                                            g // Joc.NR_COLOANE == np2 // Joc.NR_COLOANE and g % Joc.NR_COLOANE <= np2 % Joc.NR_COLOANE) and (
                                                            g // Joc.NR_COLOANE >= np1 // Joc.NR_COLOANE and g % Joc.NR_COLOANE >= np1 % Joc.NR_COLOANE):
                                                        if jo == 1:

                                                            if stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][
                                                                g % Joc.NR_COLOANE] == 'x' or \
                                                                    stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][
                                                                        g % Joc.NR_COLOANE] == '0':
                                                                listm[i1] = 'x'  # 'x'
                                                            else:
                                                                listm[i1] = '#'
                                                            i1 += 1
                                                        else:
                                                            listm[i1] = 'x'
                                                            i1 += 1

                                                    if (
                                                            g // Joc.NR_COLOANE <= np2 // Joc.NR_COLOANE and g % Joc.NR_COLOANE == np2 % Joc.NR_COLOANE) and (
                                                            g // Joc.NR_COLOANE >= np1 // Joc.NR_COLOANE and g % Joc.NR_COLOANE >= np1 % Joc.NR_COLOANE):
                                                        if dr == 1:

                                                            if stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][
                                                                g % Joc.NR_COLOANE] == 'x' or \
                                                                    stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][
                                                                        g % Joc.NR_COLOANE] == '0':
                                                                listm[i2] = 'x'
                                                            else:
                                                                listm[i2] = '#'
                                                            i2 -= 1
                                                        else:
                                                            listm[i2] = 'x'
                                                            i2 -= 1

                                                    if (
                                                            g // Joc.NR_COLOANE <= np2 // Joc.NR_COLOANE and g % Joc.NR_COLOANE <= np2 % Joc.NR_COLOANE) and (
                                                            g // Joc.NR_COLOANE == np1 // Joc.NR_COLOANE and g % Joc.NR_COLOANE >= np1 % Joc.NR_COLOANE):
                                                        if su == 1:

                                                            if stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][
                                                                g % Joc.NR_COLOANE] == 'x' or \
                                                                    stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][
                                                                        g % Joc.NR_COLOANE] == '0':
                                                                listm[i3] = 'x'
                                                            else:
                                                                listm[i3] = '#'
                                                            i3 -= 1
                                                        else:
                                                            listm[i3] = 'x'
                                                            i3 -= 1

                                                    if (
                                                            g // Joc.NR_COLOANE <= np2 // Joc.NR_COLOANE and g % Joc.NR_COLOANE <= np2 % Joc.NR_COLOANE) and (
                                                            g // Joc.NR_COLOANE >= np1 // Joc.NR_COLOANE and g % Joc.NR_COLOANE == np1 % Joc.NR_COLOANE):
                                                        if st == 1:

                                                            if stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][
                                                                g % Joc.NR_COLOANE] == 'x' or \
                                                                    stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][
                                                                        g % Joc.NR_COLOANE] == '0':
                                                                listm[i4] = 'x'
                                                            else:
                                                                listm[i4] = '#'
                                                            i4 += 1
                                                        else:
                                                            listm[i4] = 'x'
                                                            i4 += 1

                                listm = list(filter(lambda a: a != -1, listm))
                                listm.append(listm[0])
                                # print(listm)
                                for _ in range(len(listm) - 1):
                                    # print(_)
                                    if (listm[_] == 'x' and listm[_ + 1] == '#') or (
                                            listm[_] == '#' and listm[_ + 1] == 'x'):
                                        nri += 1
                                if nri > 2 or nri <= 0:
                                    test1 = 1

                            if test1 == 0:
                                for np1 in range(len(Joc.celuleGrid)):
                                    if Joc.celuleGrid[np1].collidepoint((pos_end[0], pos_start[1])):
                                        for np2 in range(len(Joc.celuleGrid)):
                                            if Joc.celuleGrid[np2].collidepoint((pos_start[0], pos_end[1])):
                                                for g in range(np1, np2 + 1):
                                                    if (
                                                            g // Joc.NR_COLOANE <= np2 // Joc.NR_COLOANE and g % Joc.NR_COLOANE <= np2 % Joc.NR_COLOANE) and (
                                                            g // Joc.NR_COLOANE >= np1 // Joc.NR_COLOANE and g % Joc.NR_COLOANE >= np1 % Joc.NR_COLOANE):
                                                        stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][
                                                            g % Joc.NR_COLOANE] = Joc.JMAX
                                                        stare_curenta.tabla_joc.ultima_mutare = (
                                                        g - np1, g % Joc.NR_COLOANE)
                                                        stare_curenta.tabla_joc.deseneaza_grid(
                                                            linie_marcaj=g // Joc.NR_COLOANE,
                                                            coloana_marcaj=g % Joc.NR_COLOANE)
                                                break
                                print("\nTabla dupa mutarea jucatorului")
                                print(str(stare_curenta))

                        if pos_start[0] > pos_end[0] and pos_start[1] > pos_end[1]:
                            test1 = 0
                            for np1 in range(len(Joc.celuleGrid)):
                                if Joc.celuleGrid[np1].collidepoint((pos_end[0], pos_end[1])) and test1 == 0:
                                    for np2 in range(len(Joc.celuleGrid)):
                                        if Joc.celuleGrid[np2].collidepoint(
                                                (pos_start[0], pos_start[1])) and test1 == 0:
                                            for g in range(np1, np2 + 1):
                                                if (
                                                        g // Joc.NR_COLOANE <= np2 // Joc.NR_COLOANE and g % Joc.NR_COLOANE <= np2 % Joc.NR_COLOANE) and (
                                                        g // Joc.NR_COLOANE >= np1 // Joc.NR_COLOANE and g % Joc.NR_COLOANE >= np1 % Joc.NR_COLOANE) and (
                                                        (stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][
                                                             g % Joc.NR_COLOANE] == 'r') or (
                                                                stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][
                                                                    g % Joc.NR_COLOANE] == '0') or (
                                                                stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][
                                                                    g % Joc.NR_COLOANE] == 'x')):
                                                    test1 = 1
                                                    print(
                                                        "Nu puteti manca zona aleasa deoarece contine cel putin o bucata otravita sau aleasa deja!")
                                                    break

                            if test1 == 0:
                                nri = 0
                                for np1 in range(len(Joc.celuleGrid)):
                                    if Joc.celuleGrid[np1].collidepoint((pos_end[0], pos_end[1])) and test1 == 0:
                                        np = np1
                                        su = 0
                                        st = 0
                                        if np1 >= Joc.NR_COLOANE:
                                            su = 1
                                            np = np - Joc.NR_COLOANE
                                        if np1 % Joc.NR_COLOANE != 0:
                                            st = 1
                                            np = np - 1
                                        np1 = np
                                        i1 = 0
                                        i2 = Joc.NR_LINII + Joc.NR_COLOANE + 5
                                        # print(str(i1) + " " + str(i2))
                                        for np2 in range(len(Joc.celuleGrid)):
                                            if Joc.celuleGrid[np2].collidepoint(pos_start[0],
                                                                                pos_start[1]) and test1 == 0:
                                                np = np2
                                                jo = 0
                                                dr = 0
                                                if np2 <= (Joc.NR_COLOANE * Joc.NR_LINII) - Joc.NR_COLOANE - 1:
                                                    jo = 1
                                                    np = np + Joc.NR_COLOANE
                                                if (np2 + 1) % Joc.NR_COLOANE != 0:
                                                    dr = 1
                                                    np = np + 1
                                                np2 = np
                                                i3 = Joc.NR_LINII + Joc.NR_COLOANE * 2 + 8
                                                i4 = Joc.NR_LINII + Joc.NR_COLOANE * 2 + 9
                                                listm = [-1 for _ in
                                                         range((Joc.NR_COLOANE + Joc.NR_LINII + 20) * 2 + 1)]
                                                for g in range(np1, np2 + 1):
                                                    if (
                                                            g // Joc.NR_COLOANE == np2 // Joc.NR_COLOANE and g % Joc.NR_COLOANE <= np2 % Joc.NR_COLOANE) and (
                                                            g // Joc.NR_COLOANE >= np1 // Joc.NR_COLOANE and g % Joc.NR_COLOANE >= np1 % Joc.NR_COLOANE):
                                                        if jo == 1:
                                                            stare_curenta.tabla_joc.deseneaza_grid(
                                                                linie_marcaj=g // Joc.NR_COLOANE,
                                                                coloana_marcaj=g % Joc.NR_COLOANE)
                                                            if stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][
                                                                g % Joc.NR_COLOANE] == 'x' or \
                                                                    stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][
                                                                        g % Joc.NR_COLOANE] == '0':
                                                                listm[i1] = 'x'  # 'x'
                                                            else:
                                                                listm[i1] = '#'
                                                            i1 += 1
                                                        else:
                                                            listm[i1] = 'x'
                                                            i1 += 1

                                                    if (
                                                            g // Joc.NR_COLOANE <= np2 // Joc.NR_COLOANE and g % Joc.NR_COLOANE == np2 % Joc.NR_COLOANE) and (
                                                            g // Joc.NR_COLOANE >= np1 // Joc.NR_COLOANE and g % Joc.NR_COLOANE >= np1 % Joc.NR_COLOANE):
                                                        if dr == 1:

                                                            stare_curenta.tabla_joc.deseneaza_grid(
                                                                linie_marcaj=g // Joc.NR_COLOANE,
                                                                coloana_marcaj=g % Joc.NR_COLOANE)
                                                            if stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][
                                                                g % Joc.NR_COLOANE] == 'x' or \
                                                                    stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][
                                                                        g % Joc.NR_COLOANE] == '0':
                                                                listm[i2] = 'x'
                                                            else:
                                                                listm[i2] = '#'
                                                            i2 -= 1
                                                        else:
                                                            listm[i2] = 'x'
                                                            i2 -= 1

                                                    if (
                                                            g // Joc.NR_COLOANE <= np2 // Joc.NR_COLOANE and g % Joc.NR_COLOANE <= np2 % Joc.NR_COLOANE) and (
                                                            g // Joc.NR_COLOANE == np1 // Joc.NR_COLOANE and g % Joc.NR_COLOANE >= np1 % Joc.NR_COLOANE):
                                                        if su == 1:
                                                            stare_curenta.tabla_joc.deseneaza_grid(
                                                                linie_marcaj=g // Joc.NR_COLOANE,
                                                                coloana_marcaj=g % Joc.NR_COLOANE)
                                                            if stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][
                                                                g % Joc.NR_COLOANE] == 'x' or \
                                                                    stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][
                                                                        g % Joc.NR_COLOANE] == '0':
                                                                listm[i3] = 'x'
                                                            else:
                                                                listm[i3] = '#'
                                                            i3 -= 1
                                                        else:
                                                            listm[i3] = 'x'
                                                            i3 -= 1

                                                    if (
                                                            g // Joc.NR_COLOANE <= np2 // Joc.NR_COLOANE and g % Joc.NR_COLOANE <= np2 % Joc.NR_COLOANE) and (
                                                            g // Joc.NR_COLOANE >= np1 // Joc.NR_COLOANE and g % Joc.NR_COLOANE == np1 % Joc.NR_COLOANE):
                                                        if st == 1:
                                                            stare_curenta.tabla_joc.deseneaza_grid(
                                                                linie_marcaj=g // Joc.NR_COLOANE,
                                                                coloana_marcaj=g % Joc.NR_COLOANE)
                                                            if stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][
                                                                g % Joc.NR_COLOANE] == 'x' or \
                                                                    stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][
                                                                        g % Joc.NR_COLOANE] == '0':
                                                                listm[i4] = 'x'
                                                            else:
                                                                listm[i4] = '#'
                                                            i4 += 1
                                                        else:
                                                            listm[i4] = 'x'
                                                            i4 += 1

                                listm = list(filter(lambda a: a != -1, listm))
                                listm.append(listm[0])
                                # print(listm)
                                for _ in range(len(listm) - 1):
                                    # print(_)
                                    if (listm[_] == 'x' and listm[_ + 1] == '#') or (
                                            listm[_] == '#' and listm[_ + 1] == 'x'):
                                        nri += 1
                                if nri > 2 or nri <= 0:
                                    test1 = 1

                            if test1 == 0:
                                for np1 in range(len(Joc.celuleGrid)):
                                    if Joc.celuleGrid[np1].collidepoint((pos_end[0], pos_end[1])):
                                        for np2 in range(len(Joc.celuleGrid)):
                                            if Joc.celuleGrid[np2].collidepoint((pos_start[0], pos_start[1])):
                                                for g in range(np1, np2 + 1):
                                                    if (
                                                            g // Joc.NR_COLOANE <= np2 // Joc.NR_COLOANE and g % Joc.NR_COLOANE <= np2 % Joc.NR_COLOANE) and (
                                                            g // Joc.NR_COLOANE >= np1 // Joc.NR_COLOANE and g % Joc.NR_COLOANE >= np1 % Joc.NR_COLOANE):

                                                        stare_curenta.tabla_joc.matr[g // Joc.NR_COLOANE][
                                                            g % Joc.NR_COLOANE] = Joc.JMAX
                                                        stare_curenta.tabla_joc.ultima_mutare = (
                                                        g - np1, g % Joc.NR_COLOANE)
                                                        stare_curenta.tabla_joc.deseneaza_grid(
                                                            linie_marcaj=g // Joc.NR_COLOANE,
                                                            coloana_marcaj=g % Joc.NR_COLOANE)
                                                break
                                print("\nTabla dupa mutarea jucatorului")
                                print(str(stare_curenta))

                        pygame.display.flip()
                        if (afis_daca_final(stare_curenta)):
                            break

                        if test1 == 0:
                            linie = np // Joc.NR_COLOANE
                            coloana = np % Joc.NR_COLOANE

                            t_dupaom = int(round(time.time() * 1000))
                            print("Jucatorul a \"gandit\" timp de " + str(
                                t_dupaom - t_inainteom) + " milisecunde.")
                            test_timp = 0
                            # afisarea starii jocului in urma mutarii utilizatorului
                            print("\nTabla dupa mutarea jucatorului")
                            print(str(stare_curenta))

                            stare_curenta.tabla_joc.deseneaza_grid(linie_marcaj=linie,
                                                                   coloana_marcaj=coloana)
                            # testez daca jocul a ajuns intr-o stare finala
                            # si afisez un mesaj corespunzator in caz ca da
                            Joc.jucator_curent = 'x'
                            if (afis_daca_final(stare_curenta)):
                                break

                            stare_curenta.j_curent = Joc.jucator_opus(stare_curenta.j_curent)
            # 13. - optiuni jucatori
            else:
                stare_curenta.j_curent = Joc.jucator_opus(stare_curenta.j_curent)

if __name__ == "__main__":
    main()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()