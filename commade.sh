#!/bin/bash

# Config
BUILD_DIR="build"
COMPUTE_CAPABILITY="61"  # Pour la Quadro P620

# Vérifier la version de CUDA
echo "🔍 Vérification de la version CUDA..."
CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -d',' -f1)
echo "✅ CUDA version détectée : $CUDA_VERSION"

# Nettoyer l'ancien build
echo "🛠 Suppression de l'ancien dossier de build..."
rm -rf $BUILD_DIR

# Créer le dossier de build
echo "📁 Création du dossier de build..."
mkdir $BUILD_DIR && cd $BUILD_DIR

# Configurer CMake
echo "⚙️ Configuration de CMake avec CC=$COMPUTE_CAPABILITY..."
cmake .. -DCC=$COMPUTE_CAPABILITY

# Compiler avec make
echo "🚀 Compilation en cours..."
make -j$(nproc)

# Vérifier si la compilation a réussi
if [ $? -ne 0 ]; then
    echo "❌ Erreur de compilation !"
    exit 1
fi

# Exécuter un test par défaut
echo "🏃 Exécution du programme..."
./sc 10 1000000 42 0 1 1  # Exécute avec des paramètres par défaut

echo "✅ Script terminé avec succès !"
