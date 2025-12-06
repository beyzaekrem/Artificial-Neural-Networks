# kullanıcıdan giriş alan + görselleştirme
import numpy as np
import matplotlib.pyplot as plt
 
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_turev(x):
    return x * (1 - x)

X = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

Y = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

ogrenme_katsayisi = 0.1
epoch_sayisi = 10000

gizli_noron_sayisi = int(input("\ngizli katmandaki noron sayisini giriniz: "))
print("\n")
np.random.seed(42)
agirlik_giris_gizli = np.random.uniform(-1, 1, (4, gizli_noron_sayisi))
agirlik_gizli_cikis = np.random.uniform(-1, 1, (gizli_noron_sayisi, 2))
bias_gizli = np.random.uniform(-1, 1, (1, gizli_noron_sayisi))
bias_cikis = np.random.uniform(-1, 1, (1, 2))

for epoch in range(epoch_sayisi):

    gizli_net = np.dot(X, agirlik_giris_gizli) + bias_gizli
    gizli_aktivasyon = sigmoid(gizli_net)
    cikis_net = np.dot(gizli_aktivasyon, agirlik_gizli_cikis) + bias_cikis
    cikis_aktivasyon = sigmoid(cikis_net)

    hata = Y - cikis_aktivasyon
    toplam_hata = np.sum(hata ** 2) / 2

    cikis_delta = hata * sigmoid_turev(cikis_aktivasyon)
    gizli_hata = cikis_delta.dot(agirlik_gizli_cikis.T)
    gizli_delta = gizli_hata * sigmoid_turev(gizli_aktivasyon)

    agirlik_gizli_cikis += gizli_aktivasyon.T.dot(cikis_delta) * ogrenme_katsayisi
    agirlik_giris_gizli += X.T.dot(gizli_delta) * ogrenme_katsayisi
    bias_cikis += np.sum(cikis_delta, axis=0, keepdims=True) * ogrenme_katsayisi
    bias_gizli += np.sum(gizli_delta, axis=0, keepdims=True) * ogrenme_katsayisi

    if epoch % 1000 == 0:
        print(f"Eğitim Turu: {epoch}, Toplam Hata: {toplam_hata:.6f}")

print("\nToplam Hata:", toplam_hata)
print("\nAğ çıktıları :\n")
print(np.round(cikis_aktivasyon, 3))

def noron_konumlari(noron_sayisi, x_degeri):
    spacing = 2 / (noron_sayisi - 1) if noron_sayisi > 1 else 1
    start_y = -1
    return [(x_degeri, start_y + i * spacing) for i in range(noron_sayisi)]

def ag_gorsellestir(giris, gizli, cikis, gizli_noron_sayisi):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(
        f"Çok Katmanlı Yapay Sinir Ağı (4-{gizli_noron_sayisi}-2 Yapısı)",
        fontsize=15, fontweight='bold', pad=45 
    )
    ax.axis("off")

    for (x1, y1) in giris:
        for (x2, y2) in gizli:
            ax.plot([x1, x2], [y1, y2], color='gray', linewidth=0.6)

    for (x1, y1) in gizli:
        for (x2, y2) in cikis:
            ax.plot([x1, x2], [y1, y2], color='gray', linewidth=0.6)

    for (x, y) in giris:
        ax.scatter(x, y, s=800, color='orange', edgecolors='black', zorder=3)
    for (x, y) in gizli:
        ax.scatter(x, y, s=800, color='green', edgecolors='black', zorder=3)
    for (x, y) in cikis:
        ax.scatter(x, y, s=800, color='blue', edgecolors='black', zorder=3)

    ax.text(-0.2, 1.25, "giris katmani (4)", fontsize=11, fontweight='bold')
    ax.text(0.8, 1.25, f"gizli katman ({gizli_noron_sayisi})", fontsize=11, fontweight='bold')
    ax.text(1.9, 1.25, "cikis katmani (2)", fontsize=11, fontweight='bold')

    plt.show()

girisler = noron_konumlari(4, 0)
gizli = noron_konumlari(gizli_noron_sayisi, 1)
cikislar = noron_konumlari(2, 2)

ag_gorsellestir(girisler, gizli, cikislar, gizli_noron_sayisi)

