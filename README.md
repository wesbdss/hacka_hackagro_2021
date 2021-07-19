# Safe Plants

Nós desenvolvemos um hardware auxiliar para detecção ervas daninhas e controle de aplicação eficiente de defensivos agrícolas na plantação.

O nosso sistema, usando cameras RGB faz identificação em tempo real de ervas daninhas em plantações em estágio vegetativo, usando um conjunto de algorítmos para controlar a aplicação destes defensivos de forma locatizada.

esse sistema é usado junto com um drone e um PixHawk 4 ( que é um sistema que fornece todas as funcionalidades básicas de controle do drone de forma automática)

como não temos o drone e nem as cameras, mostraremos o sistema por trás

O Hardware é composto de um sistema embarcado com bom processador , por exemplo, um raspberry. e duas câmeras, uma camera com sensibilidade a infravermelho e uma camera RGB. Apos a plugagem na PixHawk, o nosso sistema já está pronto para o uso.

A identificação das ervas daninhas é feita pela diferenciação da imagem espectral verde e então aplicada algoritmos de segmentação para separação. 

Para auxílio e melhores resultados, outros algoritmos utilizando NDVI visível e GLI (que são algoritmos de identificação utilizando outros fatores visuais) estão prontos para uso, entretanto a aplicação necessitaria de um processamento mais robusto.

Este é o projeto da SafePlants, obrigado por assistir.


![readme](/imagens/example.png)