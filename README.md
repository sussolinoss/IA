# IA - XOR System

## Descrizione
Questo progetto rappresenta un semplice **sistema di Intelligenza Artificiale** implementato in Java, sviluppato come **capolavoro di terza liceo scientifico**.  
Si tratta di una rete neurale che impara a risolvere il problema logico XOR, utilizzando due layer (hidden e output) con funzioni di attivazione ReLU e Sigmoid.

## Funzionalità
- Rete neurale feedforward con:
  - Input layer (2 neuroni)
  - Hidden layer (4 neuroni, attivazione ReLU)
  - Output layer (1 neurone, attivazione Sigmoid)
- Training con algoritmo di backpropagation e ottimizzazione del Binary Cross Entropy Loss
- Predizione dei valori XOR con accuratezza visibile nella console
- Codice modulare e facilmente estendibile

## Tecnologie
- Linguaggio: **Java**
- Architettura: OOP
- From Scratch

## Come testarlo?
1. Clona la repository
2. Compila ed esegui la classe `Main`
3. Guarda i risultati di training e predizione stampati in console

## Esempio di output (Training)

```
===========  Training  ===========
| Input: [0, 0] => Output: 0,5015
| Input: [0, 1] => Output: 0,5015
| Input: [1, 0] => Output: 0,9906
| Input: [1, 1] => Output: 0,0110
=================================
```

## Testing?
Verranno generati prediction di valori basati sul training del modello

## Esempio di output (Testing)

```
===========   TEST   ===========
| Input: 1.0, 0.0
| Expected: 1.0
| Result => 100.0%
=================================
```


