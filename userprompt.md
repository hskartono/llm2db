Each table requires a manually generated ID. Use the stored procedure `getAutonumber`, which takes two parameters:  
- An input parameter `@prefix`, which should be set to the table name  
- An output parameter `@result`, which will contain the generated ID.

Declare a variable to store the output parameter `@result`.

Example of calling the autonumber for the `PurchaseOrder` table:

```
DECLARE @id nvarchar(100)
EXEC getAutonumber @prefix = N'PurchaseOrder', @result = @id OUTPUT
```