# Movie Recommendation System

## Vad gör programmet?

Programmet rekommenderar fem filmer som liknar en film du väljer. Om du t.ex. skriver in "Toy Story" så hittar programmet fem andra filmer som påminner om den baserat på genre och taggar.

---

## Vilken data används?

Programmet använder två filer:
- **movies.csv** - innehåller filmtitlar och vilka genres de tillhör.
- **tags.csv** - innehåller ord som användare har skrivit om filmer.

Tillsammans ger dessa två filer en bra bild av vad varje film handlar om.

---

## Hur fungerar det?

### Steg 1 - Läs in data
Programmet laddar in de två filerna. Om filerna inte hittas skrivs ett felmeddelande ut.

### Steg 2 - Förberd  genres
Genres i datasetet ser ut såhär: Comedy|Darama|Romance.
Programmet tar bort | -tecken så att det blir: Comedy Drama Romance - alltså valiga ord separerade med mellanslag.

### Steg 3 - Förbered taggar
Alla taggar som användare har skrivit för en film slås ihop till en lång mening per film.

### Steg 4 - Kombinera genres och taggar
Genres och taggar läggs ihop till ett enda textfält per film, som sedan används för att jämföra filmer med varandra.

### Steg 5 - TF-IDF
TF-IDF är ett sätt att omvandla text till siffror som datorn kan räkna med. Det smarta med TF-IDF är att vanliga ord (t.ex. "Drama" som förekommer i tusentals filmer) får ett lågt värde, medan ovanliga ord (t.ex. "surrealism") får ett högt värde.

### Steg 6 - KNN med cosine similarity
KNN (K Nearest Neighbors) är en metod som hittar de filmer som liknar en given film mest. Cosine similarity mäter hur lika två filmer är baserat på deras text - ju högre värde, desto mer lika är de.

### Steg 7 - Rekommendationsfunktionen
Funktionen `get_recommendations()` tar emot en filmtitel, söker upp filmen i datasetet och använder KNN-modellen för att hitta de 5 mest liknande filmerna.

---

## Sammanfattning

Programmet kombinerar genres och anvämdartaggar, omvandlar de till siffror med TF-IDF och använder KNN för att hitta liknande filmer. Det är en enkel men effektiv metod som ger bättre resultat än att bara jämföra genres. 