# Attenzione
Prima di effettuare le richieste che richiedono un'interrogazione a Chroma, è necessario effettuare prima la popolazione del database, quindi bisogna caricare almeno un documento PDF. Per farlo, fare riferimento alla sezione [Caricamento di un documento nel database](#caricamento-di-un-documento-nel-database).

Le richiesta da **evitare** se ancora non è stato caricato alcun documento sono:
- [Query per la generazione](#query-per-la-generazione)
- [Eliminazione di un documento dal database](#eliminazione-di-un-documento-dal-database)
- [Recupero della lista dei documenti caricati](#recupera-lista-dei-documenti-caricati-nel-database)

Mentre quelle che possono essere effettuate sono:
- [Healthy check del Gateway](#healthy-check-del-gateway)
- [Healthy check degli altri servizi tramite il gateway](#helthy-check-degli-altri-servizi-tramite-il-gateway)


# Richieste
## GET
### Healthy check del Gateway: 
End point: http://127.0.0.1:8004 <br>
Esempio di richiesta con curl:
```shell
curl --location 'http://127.0.0.1:8004'
```

### Helthy check degli altri servizi tramite il gateway
End point: http://127.0.0.1:8004/services <br>
Esempio di richiesta con curl:
```shell
curl --location 'http://127.0.0.1:8004/services'
```

### Recupera lista dei documenti caricati nel database
End point: http://127.0.0.1:8004/documents <br>
Esempio di richiesta con curl:
```shell
curl --location 'http://127.0.0.1:8004/documents'
```

## POST
### Caricamento di un documento nel database
End point: http://127.0.0.1:8004/document <br>
Va caricato nel body un documento PDF. Da Postman selezionare il tipo di body "form-data" e inserire la chiave "file", selezionare dal menù a tendina "File" (di default è "Text") caricare il documento PDF. <br>
Esempio di richiesta con curl:
```shell
curl --location 'http://127.0.0.1:8004/document' \
--form 'file=@"path/to/file.pdf"'
```

### Query per la generazione
End point: http://127.0.0.1:8004/query <br>
Va caricato nel body la query da eseguire. Da Postman selezionare il tipo di body "raw" e inserire la query con struttura JSON. Esempio:
```json
{
    "query": "Parlami del pattern Singleton"
}
```
Esempio di richiesta con curl:
```shell
curl --location 'http://127.0.0.1:8004/query' \
--header 'Content-Type: application/json' \
--data '{
    "prompt": "Parlami del pattern Singleton"
}'
```

## DELETE
### Eliminazione di un documento dal database
End point: http://127.0.0.1:8004/document?file_name=*** <br>*** è il nome del file da eliminare. <br>
Esempio di richiesta con curl:
```shell
curl --location --request DELETE 'http://127.0.0.1:8004/document?file_name=***'
```
**Consiglio:** per evitare errori, è consigliabile recuperare la lista dei documenti caricati (facendo una richiesta all'endpoint [/document](#recupera-lista-dei-documenti-caricati-nel-database))e copiare il nome del file da eliminare.

