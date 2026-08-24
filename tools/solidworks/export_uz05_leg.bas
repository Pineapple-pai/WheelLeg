Attribute VB_Name = "ExportUZ05Leg"
Option Explicit

' Import this module into a SolidWorks VBA macro, open the leg assembly, and
' run Main. The macro only reads the active assembly and writes CSV files next
' to it under uz05_leg_export.

Private Const swDocASSEMBLY As Long = 2

Public Sub Main()
    Dim swApp As SldWorks.SldWorks
    Dim swModel As SldWorks.ModelDoc2
    Dim swAssembly As SldWorks.AssemblyDoc
    Dim outputDir As String

    Set swApp = Application.SldWorks
    Set swModel = swApp.ActiveDoc
    If swModel Is Nothing Then
        MsgBox "Open the UZ-05 leg assembly first."
        Exit Sub
    End If
    If swModel.GetType <> swDocASSEMBLY Then
        MsgBox "The active document is not an assembly."
        Exit Sub
    End If
    If swModel.GetPathName = "" Then
        MsgBox "Save the assembly before exporting."
        Exit Sub
    End If

    outputDir = Left$(swModel.GetPathName, InStrRev(swModel.GetPathName, "\")) & "uz05_leg_export"
    EnsureDirectory outputDir
    Set swAssembly = swModel

    ExportDocumentInfo swModel, outputDir & "\document.csv"
    ExportComponents swAssembly, outputDir & "\components.csv"
    ExportMates swModel, outputDir & "\mates.csv"

    MsgBox "UZ-05 leg data exported to:" & vbCrLf & outputDir
End Sub

Private Sub EnsureDirectory(ByVal path As String)
    If Dir$(path, vbDirectory) = "" Then MkDir path
End Sub

Private Function Csv(ByVal value As Variant) As String
    Dim text As String
    text = CStr(value)
    Csv = """" & Replace(text, """", """"") & """"
End Function

Private Function ArrayText(ByVal values As Variant) As String
    Dim i As Long
    Dim result As String
    If Not IsArray(values) Then
        ArrayText = ""
        Exit Function
    End If
    For i = LBound(values) To UBound(values)
        If i > LBound(values) Then result = result & ";"
        result = result & Format$(CDbl(values(i)), "0.000000000000")
    Next i
    ArrayText = result
End Function

Private Sub ExportDocumentInfo(ByVal swModel As SldWorks.ModelDoc2, ByVal outputPath As String)
    Dim handle As Integer
    handle = FreeFile
    Open outputPath For Output As #handle
    Print #handle, "assembly_path,configuration"
    Print #handle, Csv(swModel.GetPathName) & "," & Csv(swModel.ConfigurationManager.ActiveConfiguration.Name)
    Close #handle
End Sub

Private Sub ExportComponents(ByVal swAssembly As SldWorks.AssemblyDoc, ByVal outputPath As String)
    Dim components As Variant
    Dim item As Variant
    Dim component As SldWorks.Component2
    Dim transform As SldWorks.MathTransform
    Dim transformData As Variant
    Dim massData As Variant
    Dim handle As Integer

    handle = FreeFile
    Open outputPath For Output As #handle
    Print #handle, "instance,path,configuration,suppression_state,transform_4x4,mass_properties_local"

    components = swAssembly.GetComponents(False)
    If IsEmpty(components) Then
        Close #handle
        Exit Sub
    End If

    For Each item In components
        Set component = item
        transformData = Empty
        massData = Empty
        Set transform = component.Transform2
        If Not transform Is Nothing Then transformData = transform.ArrayData

        ' Component2.GetMassProperties returns COM, volume, area, mass,
        ' density, principal moments, and inertia in the referenced config.
        On Error Resume Next
        massData = component.GetMassProperties(1)
        On Error GoTo 0

        Print #handle, Csv(component.Name2) & "," & _
            Csv(component.GetPathName) & "," & _
            Csv(component.ReferencedConfiguration) & "," & _
            Csv(component.GetSuppression) & "," & _
            Csv(ArrayText(transformData)) & "," & _
            Csv(ArrayText(massData))
    Next item
    Close #handle
End Sub

Private Sub ExportMates(ByVal swModel As SldWorks.ModelDoc2, ByVal outputPath As String)
    Dim feature As SldWorks.Feature
    Dim child As SldWorks.Feature
    Dim mate As SldWorks.Mate2
    Dim entity As SldWorks.MateEntity2
    Dim entityParams As Variant
    Dim entityIndex As Long
    Dim entityCount As Long
    Dim componentName As String
    Dim handle As Integer

    handle = FreeFile
    Open outputPath For Output As #handle
    Print #handle, "mate_name,mate_type,alignment,entity_index,component,entity_params,error_code"

    Set feature = swModel.FirstFeature
    Do While Not feature Is Nothing
        If feature.GetTypeName2 = "MateGroup" Then
            Set child = feature.GetFirstSubFeature
            Do While Not child Is Nothing
                Set mate = Nothing
                On Error Resume Next
                Set mate = child.GetSpecificFeature2
                On Error GoTo 0
                If Not mate Is Nothing Then
                    entityCount = 0
                    On Error Resume Next
                    entityCount = mate.GetMateEntityCount
                    On Error GoTo 0
                    If entityCount = 0 Then
                        Print #handle, Csv(child.Name) & "," & Csv(mate.Type) & "," & _
                            Csv(mate.Alignment) & ",,,," & Csv(child.GetErrorCode2)
                    Else
                        For entityIndex = 0 To entityCount - 1
                            Set entity = Nothing
                            entityParams = Empty
                            componentName = ""
                            On Error Resume Next
                            Set entity = mate.MateEntity(entityIndex)
                            If Not entity Is Nothing Then
                                entityParams = entity.EntityParams
                                If Not entity.ReferenceComponent Is Nothing Then _
                                    componentName = entity.ReferenceComponent.Name2
                            End If
                            On Error GoTo 0
                            Print #handle, Csv(child.Name) & "," & Csv(mate.Type) & "," & _
                                Csv(mate.Alignment) & "," & Csv(entityIndex) & "," & _
                                Csv(componentName) & "," & Csv(ArrayText(entityParams)) & "," & _
                                Csv(child.GetErrorCode2)
                        Next entityIndex
                    End If
                End If
                Set child = child.GetNextSubFeature
            Loop
        End If
        Set feature = feature.GetNextFeature
    Loop
    Close #handle
End Sub
