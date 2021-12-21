/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package library.management;
import java.sql.*;
//all interfaces for java

/**
 *
 * @author ameya
 */
//Rhis conn class is used to have connection of database with java program
public class conn {
//    creating a contructor
    Connection c;// connection for connecting to a database
    Statement s;// creating statement for executing queries 
    public conn(){
        try {
            // importing a class for jdbc connection
            Class.forName("com.mysql.jdbc.Driver");
//            Class.forName("com.mysql.cj.jdbc.Driver");
            // create connection to database
              c = DriverManager.getConnection("jdbc:mysql://library_managementdb","root","asggb222");
            // execute a statement or query
              s = c.createStatement();
              
        }        catch (Exception e) {
            System.out.println(e);            
        }
}
    
}
