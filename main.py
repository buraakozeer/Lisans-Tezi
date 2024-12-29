import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def goruntu_ozellikleri(goruntu_yolu):
    """
    Görüntüden özellik çıkarır.
    """
    # Görüntüyü oku
    goruntu = cv2.imread(str(goruntu_yolu))
    if goruntu is None:
        raise ValueError(f"Görüntü okunamadı: {goruntu_yolu}")
    
    # Görüntüyü yeniden boyutlandır
    goruntu = cv2.resize(goruntu, (200, 200))
    
    # HSV dönüşümü
    hsv = cv2.cvtColor(goruntu, cv2.COLOR_BGR2HSV)
    
    # Renk özellikleri
    ortalama_hsv = cv2.mean(hsv)
    
    # Kenar özellikleri
    gri = cv2.cvtColor(goruntu, cv2.COLOR_BGR2GRAY)
    kenarlar = cv2.Canny(gri, 100, 200)
    kenar_yogunlugu = np.sum(kenarlar) / (200 * 200)
    
    # Doku özellikleri (GLCM)
    gri = cv2.GaussianBlur(gri, (7, 7), 0)
    glcm = np.zeros((256, 256), dtype=np.uint32)
    for i in range(gri.shape[0]-1):
        for j in range(gri.shape[1]-1):
            i_intensity = gri[i, j]
            j_intensity = gri[i, j+1]
            glcm[i_intensity, j_intensity] += 1
            
    glcm = glcm / np.sum(glcm)
    contrast = np.sum(np.square(np.arange(256)[:, np.newaxis] - np.arange(256)) * glcm)
    
    # Tüm özellikleri birleştir
    ozellikler = np.array([
        ortalama_hsv[0], ortalama_hsv[1], ortalama_hsv[2],
        kenar_yogunlugu,
        contrast
    ])
    
    return ozellikler

def veri_seti_olustur(saglikli_klasor, sagliksiz_klasor):
    """
    Sağlıklı ve sağlıksız yaprak görüntülerinden veri seti oluşturur.
    """
    X = []  # özellikler
    y = []  # etiketler
    paths = []  # görüntü yolları
    
    # Sağlıklı yapraklar
    saglikli_yol = Path(saglikli_klasor)
    for goruntu_yolu in saglikli_yol.glob('*'):
        if goruntu_yolu.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            try:
                ozellikler = goruntu_ozellikleri(goruntu_yolu)
                X.append(ozellikler)
                y.append(1)  # 1: sağlıklı
                paths.append(str(goruntu_yolu))
            except Exception as e:
                print(f"Hata ({goruntu_yolu}): {str(e)}")
    
    # Sağlıksız yapraklar
    sagliksiz_yol = Path(sagliksiz_klasor)
    for goruntu_yolu in sagliksiz_yol.glob('*'):
        if goruntu_yolu.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            try:
                ozellikler = goruntu_ozellikleri(goruntu_yolu)
                X.append(ozellikler)
                y.append(0)  # 0: sağlıksız
                paths.append(str(goruntu_yolu))
            except Exception as e:
                print(f"Hata ({goruntu_yolu}): {str(e)}")
    
    return np.array(X), np.array(y), np.array(paths)

def model_egit(X, y, paths):
    """
    Özellikler, etiketler ve yollar kullanarak SVM modelini eğitir.
    """
    X_train, X_test, y_train, y_test, paths_train, paths_test = train_test_split(X, y, paths, test_size=0.2, random_state=42)
    
    model = SVC(kernel='rbf', probability=True)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    print("\nSınıflandırma Raporu:")
    print(classification_report(y_test, y_pred, target_names=['Sağlıksız', 'Sağlıklı']))
    
    print("\nKarmaşıklık Matrisi:")
    print(confusion_matrix(y_test, y_pred))
    
    return model, X_test, y_test, paths_test

def goruntu_siniflandir(model, goruntu_yolu):
    """
    Tek bir görüntüyü sınıflandırır.
    """
    try:
        ozellikler = goruntu_ozellikleri(goruntu_yolu)
        tahmin = model.predict([ozellikler])[0]
        olasiliklar = model.predict_proba([ozellikler])[0]
        
        goruntu = cv2.imread(str(goruntu_yolu))
        goruntu = cv2.resize(goruntu, (400, 400))
        
        
        durum = "Sağlıklı" if tahmin == 1 else "Sağlıksız"
        guven = olasiliklar[1] if tahmin == 1 else olasiliklar[0]
        
        cv2.putText(goruntu, f"Durum: {durum}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if tahmin == 1 else (0, 0, 255), 2)
        cv2.putText(goruntu, f"Güven: {guven:.2f}", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if tahmin == 1 else (0, 0, 255), 2)
    
                
        
        return goruntu, durum, guven
        
    except Exception as e:
        print(f"Hata: {str(e)}")
        return None, None, None

def main():
    saglikli_klasor = "train/healthy"
    sagliksiz_klasor = "train/diseased"
    
    try:
        print("Veri seti oluşturuluyor...")
        X, y, paths = veri_seti_olustur(saglikli_klasor, sagliksiz_klasor)
        
        if len(X) == 0:
            print("Hiç görüntü bulunamadı!")
            return
        
        print(f"Toplam {len(X)} görüntü işlendi")
        print(f"Sağlıklı: {np.sum(y == 1)}")
        print(f"Sağlıksız: {np.sum(y == 0)}")
        
        
        print("\nModel eğitiliyor...")
        model, X_test, y_test, paths_test = model_egit(X, y, paths)
        
        joblib.dump(model, 'yaprak_siniflandirici.joblib')
        print("\nModel kaydedildi: yaprak_siniflandirici.joblib")
        
        print("\nÖrnek test sonuçları:")
        for i in range(min(5, len(paths_test))):  
            goruntu, durum, guven = goruntu_siniflandir(model, paths_test[i])
            if goruntu is not None:
                print(f"Test {i+1}: Gerçek: {'Sağlıklı' if y_test[i] == 1 else 'Sağlıksız'}, "
                      f"Tahmin: {durum}, Güven: {guven:.2f}")
            
    
    except Exception as e:
        print(f"Program hatası: {str(e)}")

if __name__ == "__main__":
    main()