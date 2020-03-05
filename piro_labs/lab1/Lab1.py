
# coding: utf-8

# ## NumPy - co trzeba wiedzieć

# * Jak stworzyć tablicę (1D/2D/3D):
#   * wypełnioną zerami, jedynkami
#   * wypełnioną danymi podanymi jako lista pythona

# In[ ]:


import numpy as np
 
# tworzenie tablicy wypelnionej zerami
img = np.zeros((3,5)) # 3 wiersze, 5 kolumn
print(img)


# In[ ]:


img = np.ones((3,5), dtype=np.uint8) # wymuś typ byte (0-255)
print(img)


# In[ ]:


img = np.array([[11,12,13],[21,22,23]])
print(img)


# In[ ]:


# tworzenie tablicy wypelnionej kolejnymi liczbami
img = np.arange(10)
print(img)


# * `tablica.shape` — wymiary tablicy
# * `tablica.dtype` — typ elementów w tablicy

# In[ ]:


img = np.array([[11,12,13],[21,22,23]])
print(img)
print(img.shape)
print(img.dtype)
print(img.astype(float))  # konwersja typu elementów tablicy


# * Jak zaindeksować pojedynczy element tablicy (dla obrazów: **najpierw nr wiersza** potem nr kolumny).
# * Jak odwoływać się do zakresów: `tablica[start_włącznie:koniec_bez:krok]`

# In[ ]:


# fragment tablicy - przypadek 1D
img = np.arange(10)
print(img[1:9])     # zakres od indeksu 1 włącznie do 9 (z wyłączeniem 9)
print(img[0:10:2])  # co druga wartość
print(img[10:0:-1]) # z krokiem -1 (czyli od tył), UWAGA: element o indeksie 0 nie należy do wyniku!
print(img[10::-1])  # z krokiem -1 (czyli od tył),
print(img[::-1])    # wynik jak wyżej (cały zakres, z krokiem -1)


# In[ ]:


# ujemne indeksy: liczymy od końca
print(img)
print(img[0:-2])


# In[ ]:


# tworzenie tablicy 2D wypelnionej kolejnymi liczbami
img = np.arange(15).reshape((3,5))
print(img)
print(img.shape) # wymiary tablicy


# In[ ]:


# fragment tablicy - przypadek 2D
print(img[1,3])      # pojedynczy element z wiersza(!) o indeksie 1 i kolumnie o indeksie 3
print(img[1:3])      # nie indeksujemy kolumn (ogólnie: jakiegoś wymiaru) więc interesują nas wszystkie
print(img[1:3,:])    # jak wyżej, jawnie podany cały zakres
print(img[1:3,2])    # konkretny wiersz, UWAGA: wynik jest tablicą o mniejszej liczbie wymiarów (tutaj: wektor 1D)!
print(img[1:3,2:3])  # zakres wiersz - nie następuje redukcja wymiarów (wynik 2D)
print(img[1:3,::2])  # dla każdy wymiaru można podawać dowolny zakres


# * Odwołanie się do fragmentu (lub całości) jakiejś tablicy nie powoduje utworzenia kopii danych! Czyli modyfikując wycięty fragment tablicy modyfikujemy wybrane elementy oryginalnej tablicy.

# In[ ]:


# kopiowanie
a = np.arange(5)
b = a
c = a.copy()
print(a,b,c)
b[2] = 8
print(a,b,c)


# In[ ]:


img = np.arange(15).reshape((3,5))
print(img)
fragment = img[1,2:4]
fragment[:] = -8
print(img)
print(fragment)


# * Standardowe operacje arytmetyczne jak dodawanie i mnożenie(!) na argumentach będących tablicami wykonują operacje na odpowiadających sobie elementach. Podobnie – funkcje takie jak np.sin obliczają swoją wartość dla każdego elementu podanej tablicy.
#   **Uwaga!** W szczególności mnożenie (*) nie jest mnożeniem macierzowym!

# In[ ]:


# prosta arytmetyka
a = np.arange(5)
b = a * 2
print(a, b)
 
c = a + b
print(c)
print(a + b - c)


# In[ ]:


# operacje logiczne
tab = np.arange(10)
print(tab < 5)
print(tab == 5)
print(np.logical_or(tab < 2, tab > 6))
print((tab < 2) | (tab > 6)) # alternatywny zapis - wynik jak wyżej (proszę zwrócić uwagę na nawiasy!)


# * Funkcje takie jak `np.sum` czy `np.max` wyliczają wartość z całej tablicy, lub po określonej osi.

# In[ ]:


img = np.arange(15).reshape((3,5))
print(img)
print(np.sum(img), img.sum())
print(np.mean(img, axis=0))
print(np.min(img, axis=1))


# * Jak odwoływać się do wybranych elementów: `tablica[boolowska_tablica]` (np. spełniających kreślony warunek: `tablica[tablica < 10]`)

# In[ ]:


tab = np.arange(10)
# cond = np.logical_and(tab >= 2, tab <= 7)
cond = (tab >= 2) & (tab <= 7)
print(cond)
print(tab[cond])


# # Obrazki
# 
# Przygotowania...

# In[ ]:


# poniższa linijka mówi jupyter'owi żeby obrazki wyświetlał tutaj jako wynik wykonania komórki
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import skimage.data as data
import skimage.color as color
import skimage


# Wczytaj i wyświetl przykładowy obrazek:

# In[ ]:


# opcjonalnie można ustawić dpi i wielkość obrazka w calach
import matplotlib as mpl
mpl.rc("savefig", dpi=100)
plt.figure(figsize=(10, 10))

img = data.astronaut()
#img = io.imread('plik na dysku lub adres obrazka w sieci.png')
io.imshow(img)


# In[ ]:


io.imshow(img[50:200,150:300]) # Wycięcie fragmentu obrazka


# In[ ]:


io.imshow(color.rgb2grey(img)) # Konwersja obrazku do odcieni szarości 

# io.imread ma parametr as_grey do konwersji przy wczytywaniu
# img = io.imread('plik na dysku lub adres obrazka w sieci.png', as_grey=True)


# Alternatywne wyświetlenie obrazka w zewnętrznym programie:
# ##### UWAGA! Musisz zamknąć zewnętrzny program aby kod dalej się wykonywał

# In[ ]:


from scipy.misc import toimage
toimage(img).show()
print("KONIEC")


# # Ćwiczenia:
# * Napisz funkcję zliczającą liczbę różniących się pikseli pomiędzy dwoma obrazkami (zakładamy taki sam rozmiar obrazków).
#   
#   **Uwaga:** obrazki mogą być albo kolorowe (3 kanały: tablica o wymiarach `width x height x 3`) albo jednokanałowe (tablica o wymiarach `width x height`)
#   
#   **Uwaga ogólna:** gdy mówimy o obrazie **jednokanałowym** mamy na myśli tablicę o rozmiarze:
#     * `width x height` (dwa wymiary)
#     * a **NIE** `width x height x 1` (trzy wymiary)! 

# In[ ]:


def cmpDiff(ground_true, im_test):
    return # TODO


# In[ ]:


img1 = data.astronaut()
img2 = img1.copy()
img2[10,10] = (0,255,255)
print(cmpDiff(img1, img2)) # == 1
assert cmpDiff(img1, img2) == 1


# * Napisz funkcję porównującą dwa obrazki jednokanałowe interpretowane **binarnie** (punkty mają wartość 0 lub różną od zera) i zwracającą dwie wartości:
#  1. false positive – powinno być zero a nie jest,
#  2. false negative – jest zero a nie powinno być.

# In[ ]:


def cmpTPTN(im_true, im_test):
    return 0, 0  # TODO


# * Przetestuj działanie rozmycia gaussowskiego:

# In[ ]:


import ipywidgets
img = data.camera()
import skimage.filters

sigma_slider = ipywidgets.FloatSlider(min=0, max=50, step=1, value=1)
@ipywidgets.interact(sigma=sigma_slider)
def gaussian_demo(sigma=5):
    io.imshow(skimage.filters.gaussian(img, sigma))


# * Postaraj się wykryć obiekt(y) w obrazkach (abc i uniform to obrazki które należy wczytać z dysku):
#   1. task1 = abc + (uniform - 127)
#   2. task2 = abc + 1.1*(uniform - 127)
#   3. task3 = abc + 1.5*(uniform - 127)
# 
# Przez wykrycie należy rozumieć wygenerowanie obrazka-maski w której w miejscach gdzie jest obiekt mamy wartość większą od 0, a zera w miejscach gdzie nie ma obiektu. Do porównywania swoich wyników wykorzystaj funkcje `cmpDiff`, `cmpTPTN` z poprzednich ćwiczeń.
# 
# Czy udało Ci się zminimalizować różnicę do zera?
#   
# Przydatne funkcje, np.: `gaussian`, `median`
#   
# **Uwaga:** Jaki typ ma tablica `abc` i `uniform`? Zwróć uwagę co się dzieje, gdy wartości wychodzą poza oryginalny zakres 0-255. Wykonując operacje artymetyczne przejdź na odpowiedni typ danych, a następnie przytnij wartości do zakrsu 0-255 (np.clip) i wróć do typu `np.uint8`.

# In[ ]:


abc = io.imread('components/objects/abc.png', as_grey=True)
uniform = io.imread('components/noise/uniform.png', as_grey=True)
# task1 = abc + (uniform - 127)
# task2 = abc + 1.1*(uniform - 127)
# task3 = abc + 1.5*(uniform - 127)


# * Wykryj obiekty w obrazku: 0.2 \* abc + 0.8 \* circular.
#   
#   Podobne obrazki: [PiRO](http://www.cs.put.poznan.pl/bwieloch/PiRO/piro_grey.png) oraz [zaszumione PiRO](http://www.cs.put.poznan.pl/bwieloch/PiRO/piro_noise_grey.png).
#   
#   Potencjalnie przydatne funkcje: `threshold_adaptive`
