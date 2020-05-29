# ArkNightsDataServer
Warning: This project is in development.

##Keywords
- Arknights
- Full text search

## Introduce

### What does this program do?
- Load data from [ArknightsGameData](https://github.com/Kengxxiao/ArknightsGameData)
- Search game characters by tags statistics and skill feature.
- Search any text in the game.
- Filter search result by characters the user owns.
- Load characters the user owns from screenshots.
- 
- Expose RESTful API

Elastic search is used for full text(such as character lines or plot texts) search and tag search.
As for search with characters' certain skill features or statistics, database backend is the better choice.

## TODOs

1. OCR to load box.
2. Refactor TAGS.
