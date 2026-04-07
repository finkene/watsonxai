# Entity Relationship Diagram (ERD)

Generated from DDL files in `/Users/finkene/Documents/data/sample_data/Synthetic_Data/test_ddl`

## Database Schema: sdg212

This ERD represents a financial transaction and fraud detection system with the following entities:

```mermaid
erDiagram
    users ||--o{ cards : "has"
    users ||--o{ trans : "makes"
    users ||--o{ liquid_accts_people : "owns"
    
    cards ||--o{ trans : "used_in"
    
    banks ||--o{ bank_xfers : "from_bank"
    banks ||--o{ bank_xfers : "to_bank"
    banks ||--o{ liquid_accts_people : "holds"
    banks ||--o{ liquid_accts_companies : "holds"
    
    b2b ||--o{ liquid_accts_companies : "owns"
    
    users {
        INTEGER Person_Index
        VARCHAR Entity_ID PK
        VARCHAR Person
        INTEGER Start_Age
        INTEGER End_Age
        INTEGER Retirement_Age
        DATE Birth_Date
        VARCHAR Gender
        VARCHAR Address
        VARCHAR Apartment
        VARCHAR City
        VARCHAR State
        VARCHAR Postal_Code
        VARCHAR Country
        DOUBLE Latitude
        DOUBLE Longitude
        VARCHAR Currency
        INTEGER Is_Criminal
        INTEGER Per_Capita_Income_Postal_Code
        INTEGER Yearly_Income_Person
        INTEGER Total_Initial_Debt
        INTEGER FICO_Score
        INTEGER Num_Card_Accounts
    }
    
    cards {
        INTEGER User PK_FK
        VARCHAR Entity_ID
        INTEGER Card_Index PK
        VARCHAR Card_Brand
        VARCHAR Card_Type
        VARCHAR Card_Currency
        BIGINT Card_Number
        VARCHAR Financial_Institution_ID
        BIGINT Account_ID
        VARCHAR Initial_Expiration_Date
        INTEGER CVV
        VARCHAR Has_Chip
        VARCHAR Has_Tap
        INTEGER Cards_Issued
        DOUBLE Initial_Balance
        DOUBLE Final_Balance
        DOUBLE Credit_Limit
        VARCHAR Acct_Open_Date
        VARCHAR Last_Fraudulent_Use
        INTEGER Year_Pin_Last_Changed
        VARCHAR Card_on_Dark_Web
        INTEGER Lifetime_Transactions
        INTEGER Fraudulent_Transactions
        INTEGER Mean_Transactions_per_Month
    }
    
    trans {
        INTEGER User FK
        VARCHAR Entity_ID
        INTEGER Card FK
        DATE Transaction_Date
        VARCHAR Transaction_Day_of_Week
        VARCHAR Transaction_Time
        VARCHAR Transaction_Ref_ID
        DOUBLE Payment_to_Merchant
        VARCHAR Merchant_Currency
        FLOAT Charge_to_Buyer
        VARCHAR Buyer_Currency
        VARCHAR Method
        VARCHAR Merchant_Name
        VARCHAR Merchant_ID
        VARCHAR Merchant_Location_ID
        VARCHAR Merchant_City
        VARCHAR Merchant_State
        VARCHAR Postal_Code
        VARCHAR Country
        DOUBLE Latitude
        DOUBLE Longitude
        INTEGER MCC
        BOOLEAN Is_Online
        BOOLEAN Is_Hold
        BOOLEAN Is_Flight
        DATE Flight_Date
        VARCHAR Flt1_Src_Airport
        VARCHAR Flt1_Dest_Airport
        VARCHAR Flt2_Src_Airport
        VARCHAR Flt2_Dest_Airport
        BOOLEAN Is_Fraud
        VARCHAR Fraudster_ID
        VARCHAR IBM_Internal
        VARCHAR Errors
    }
    
    banks {
        VARCHAR Bank_ID PK
        VARCHAR Bank_Name
        INTEGER Num_Transactions
        INTEGER Num_Total_Locations
        INTEGER Num_non_Focus_Country_Locns
        VARCHAR Sample_City
    }
    
    bank_xfers {
        BIGINT Transaction_Number PK
        VARCHAR Transaction_Date
        VARCHAR Transaction_Time
        VARCHAR Transaction_Day_of_Week
        VARCHAR From_Bank FK
        BIGINT From_Account
        VARCHAR To_Bank FK
        BIGINT To_Account
        DOUBLE Amount_Paid
        VARCHAR Payment_Currency
        DOUBLE Amount_Received
        VARCHAR Receiving_Currency
        VARCHAR Payment_Format
        DOUBLE From_Initial_Balance
        DOUBLE From_End_Balance
        DOUBLE To_Initial_Balance
        DOUBLE To_End_Balance
        VARCHAR Transaction_Type
        VARCHAR Laundering_Type
        BOOLEAN Is_Laundering
        BOOLEAN Is_Cheque_Fraud
        BOOLEAN Is_APP_Fraud
        VARCHAR Cheque_Fraudster_ID
        VARCHAR APP_Fraudster_ID
        INTEGER APP_Fraud_Sequence_Number
        BOOLEAN Sufficient_Funds
        BOOLEAN Overdraft_Okay
        BOOLEAN Is_All_Cash
        BOOLEAN Is_Hold
    }
    
    b2b {
        INTEGER Company_Index
        VARCHAR Company_ID PK
        VARCHAR Company_Name
        INTEGER Company_MCC
        INTEGER USD_Val_of_B2B_Payments_Recvd
        INTEGER USD_Val_of_B2B_Payments_Made
        INTEGER Number_of_B2B_Payments_Recvd
        INTEGER Number_of_B2B_Payments_Made
        INTEGER Number_of_B2B_Payments_Missed
        INTEGER Number_of_Main_Suppliers
        INTEGER Number_of_Main_B2B_Customers
        INTEGER Index_Number_of_Supplier
        VARCHAR Supplier_Company_ID
        VARCHAR Supplier_Company_Name
        INTEGER Supplier_Company_MCC
        VARCHAR Supplier_Size_Category
        INTEGER Supplier_Expctd_USD_Sell_to_Co
        INTEGER Supplier_Actual_USD_Sell_to_Co
        INTEGER Index_Number_of_B2B_Customer
        VARCHAR B2B_Customer_Company_ID
        VARCHAR B2B_Customer_Company_Name
        INTEGER B2B_Customer_Company_MCC
        VARCHAR B2B_Customer_Size_Category
        INTEGER B2B_Cust_Expctd_USD_Buys_fr_Co
        INTEGER B2B_Cust_Actual_USD_Buys_fr_Co
    }
    
    liquid_accts_people {
        VARCHAR Financial_Institution_Name
        VARCHAR Financial_Institution_ID FK
        VARCHAR Branch
        VARCHAR Account_Country
        VARCHAR Account_Currency
        VARCHAR Entity_Type
        VARCHAR Entity_ID PK_FK
        VARCHAR Entity_Name
        BOOLEAN Does_Account_Have_Debit_Card
        INTEGER DebitCard_Index_in_EntityCards
        BOOLEAN Controlled_by_Criminal
        INTEGER MCC
        VARCHAR Account_Type
        BIGINT Account_ID PK
        INTEGER Max_Overdraft
    }
    
    liquid_accts_companies {
        VARCHAR Financial_Institution_Name
        VARCHAR Financial_Institution_ID FK
        VARCHAR Branch
        VARCHAR Account_Country
        VARCHAR Account_Currency
        VARCHAR Entity_Type
        VARCHAR Entity_ID PK_FK
        VARCHAR Entity_Name
        BOOLEAN Does_Account_Have_Debit_Card
        INTEGER DebitCard_Index_in_EntityCards
        BOOLEAN Controlled_by_Criminal
        INTEGER MCC
        VARCHAR Account_Type
        BIGINT Account_ID PK
        INTEGER Max_Overdraft
    }
```

## Table Descriptions

### Core Entities

1. **users** - Individual persons/customers in the system
   - Primary Key: `Entity_ID`
   - Contains personal information, demographics, and financial profile

2. **cards** - Credit/debit cards issued to users
   - Composite Primary Key: `(User, Card_Index)`
   - Links to users and tracks card details and fraud indicators

3. **trans** - Card transactions made by users
   - Foreign Keys: `User`, `Card`
   - Tracks all transaction details including fraud flags

4. **banks** - Financial institutions
   - Primary Key: `Bank_ID`
   - Contains bank information and location data

5. **bank_xfers** - Bank-to-bank transfers
   - Primary Key: `Transaction_Number`
   - Tracks inter-bank transfers with fraud detection flags

6. **b2b** - Business-to-business relationships and payments
   - Primary Key: `Company_ID`
   - Contains company payment relationships and supplier/customer data

7. **liquid_accts_people** - Liquid accounts owned by individuals
   - Composite Primary Key: `(Entity_ID, Account_ID)`
   - Links people to their bank accounts

8. **liquid_accts_companies** - Liquid accounts owned by companies
   - Composite Primary Key: `(Entity_ID, Account_ID)`
   - Links companies to their bank accounts

## Key Relationships

- **Users to Cards**: One-to-many (a user can have multiple cards)
- **Users to Transactions**: One-to-many (a user makes multiple transactions)
- **Cards to Transactions**: One-to-many (a card is used in multiple transactions)
- **Banks to Bank Transfers**: One-to-many (banks participate in transfers as sender/receiver)
- **Users to Liquid Accounts (People)**: One-to-many (a person can have multiple accounts)
- **Companies to Liquid Accounts (Companies)**: One-to-many (a company can have multiple accounts)
- **Banks to Liquid Accounts**: One-to-many (banks hold multiple accounts)

## Fraud Detection Features

The schema includes multiple fraud detection indicators:
- `Is_Fraud` in transactions
- `Is_Laundering`, `Is_Cheque_Fraud`, `Is_APP_Fraud` in bank transfers
- `Fraudster_ID` fields for tracking fraudulent actors
- `Card_on_Dark_Web` indicator in cards table
- `Is_Criminal` flag in users table
- `Controlled_by_Criminal` in liquid accounts

## Notes

- Schema name: `sdg212`
- Some tables have data quality issues noted in DDL comments (e.g., non-unique primary keys in b2b)
- Date/time fields use various formats (DATE, VARCHAR) - consider standardization
- Currency fields present for multi-currency support
- Geographic data (latitude/longitude) available for location-based analysis